"""CM 的真正瓶颈：不是波形，是 57 人 × 29 个共线特征的维度灾难。

依据
----
12 号勘误记录：CM 清洗后段级线性上限 0.157，人级 LOOCV 0.080，但**单量 h1 = 0.392**。
57 个受试者、每人一个 SBP；29 个物理量的相关矩阵条件数 1e13（18 号 §7）。
在这个配比下多元回归必然崩溃——不是没信号，是信号被方差淹没。

本脚本用严格嵌套交叉验证，比较"降维/选量"能否把 ρ 从 0.08 拉到 0.35+。

必须嵌套，否则数字无意义
------------------------
57 个样本上，若在全体上挑"最好的那个量"，ρ 会被选择偏差抬高 0.1 以上
（57 个噪声变量里最大相关约 ±0.26，p<0.05 的假阳性必然出现）。
因此外层留一人，内层在训练折里选超参/选量，被留出的人完全不见于任何选择步骤。

比较的四个模型
--------------
  ridge29   29 量 + 岭回归（内层 5 折选 α）
  pls      偏最小二乘（内层选 1–4 成分）
  single   单量（内层从 29 量里挑 |r| 最大者）—— 带选择偏差，但内层选所以合法
  h1only   固定用 h1，不做任何选择（最保守的参照）

全部按受试者聚合到人级（每人的段特征取中位），标签为每人一个 SBP。
报告 ρ 与 95% 自助区间（对 57 个预测值重抽，2000 次）。
"""
from __future__ import annotations

import os as _os, sys as _sys

# Resolved at run time: scripts/ holds sibling groups, the repository root holds data/.
_S = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _S)
for _d in _os.listdir(_S):
    if _d[:1].isdigit() and _os.path.isdir(_os.path.join(_S, _d)):
        _sys.path.append(_os.path.join(_S, _d))
import _repo  # noqa: E402


import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_mediator as PM
import phys_layer as PL

RNG = np.random.default_rng(0)


def rank(u, v):
    ok = np.isfinite(u) & np.isfinite(v)
    if ok.sum() < 5:
        return np.nan
    ru = np.argsort(np.argsort(u[ok])).astype(float); ru -= ru.mean()
    rv = np.argsort(np.argsort(v[ok])).astype(float); rv -= rv.mean()
    d = np.sqrt((ru ** 2).sum() * (rv ** 2).sum())
    return float((ru * rv).sum() / d) if d > 0 else np.nan


def ridge_fit(X, y, alphas):
    """返回各 alpha 的系数（含截距），X 已标准化。"""
    n, p = X.shape
    out = []
    for a in alphas:
        A = X.T @ X + a * np.eye(p)
        b = np.linalg.solve(A, X.T @ (y - y.mean()))
        out.append((b, y.mean()))
    return out


def pls_fit(X, y, nc):
    """NIPALS，返回可用于预测的 (W, P, Q) 简化实现。"""
    Xc = X - X.mean(0); yc = y - y.mean()
    W, P, Q = [], [], []
    for _ in range(nc):
        w = Xc.T @ yc
        w /= (np.linalg.norm(w) + 1e-12)
        t = Xc @ w
        p = (Xc.T @ t) / (t @ t + 1e-12)
        q = (yc @ t) / (t @ t + 1e-12)
        Xc = Xc - np.outer(t, p)
        yc = yc - q * t
        W.append(w); P.append(p); Q.append(q)
    return W, P, Q, X.mean(0), y.mean()


def pls_pred(model, X):
    W, P, Q, xm, ym = model
    Xc = X - xm
    y = np.full(len(X), ym)
    for w, p, q in zip(W, P, Q):
        t = Xc @ w
        Xc = Xc - np.outer(t, p)
        y = y + q * t
    return y


def inner_cv(X, y, alphas, nfold=5):
    """内层 5 折：分别给 ridge / pls 选超参，返回预测函数。"""
    n = len(y)
    idx = RNG.permutation(n)
    folds = np.array_split(idx, nfold)

    # ridge：取平均 CV 误差最小的 alpha
    err = np.zeros(len(alphas))
    for f in folds:
        tr = np.setdiff1d(idx, f)
        Xtr = (X[tr] - X[tr].mean(0)) / (X[tr].std(0) + 1e-12)
        Xte = (X[f] - X[tr].mean(0)) / (X[tr].std(0) + 1e-12)
        for j, (b, mu) in enumerate(ridge_fit(Xtr, y[tr], alphas)):
            err[j] += ((Xte @ b + mu - y[f]) ** 2).sum()
    a_best = alphas[int(np.argmin(err))]

    # pls：取平均 CV 误差最小的成分数
    errp = np.zeros(4)
    for f in folds:
        tr = np.setdiff1d(idx, f)
        for c in range(1, 5):
            m = pls_fit(X[tr], y[tr], c)
            errp[c - 1] += ((pls_pred(m, X[f]) - y[f]) ** 2).sum()
    c_best = int(np.argmin(errp)) + 1

    # single：内层按 |r| 选量，**并按训练集学到的符号预测**
    # 不取符号会出错：内层 280 折里 260 次选 h1（与 SBP 正）、20 次选 kurt（负），
    # 直接把变量值当预测值会让两种方向互相抵消，把 0.389 压成 0.313。
    Xc = (X[tr := np.arange(n)] - X.mean(0)) / (X.std(0) + 1e-12)
    rs = np.array([rank(Xc[:, j], y) for j in range(X.shape[1])])
    j_best = int(np.nanargmax(np.abs(rs)))
    sign = 1.0 if rs[j_best] >= 0 else -1.0

    def pred_ridge(Xte):
        mu, sd = X.mean(0), X.std(0) + 1e-12
        Xn = (Xte - mu) / sd
        b, m0 = ridge_fit((X - mu) / sd, y, [a_best])[0]
        return Xn @ b + m0

    def pred_pls(Xte):
        return pls_pred(pls_fit(X, y, c_best), Xte)

    def pred_single(Xte):
        return sign * Xte[:, j_best]

    return dict(ridge=pred_ridge, pls=pred_pls, single=pred_single,
                alpha=a_best, ncomp=c_best, jpick=j_best)


def main():
    os.environ.setdefault("CM_VER", "v3")
    d = PM.load_source("CM")
    X, y, g = np.asarray(d["X"], float), np.asarray(d["y"], float), np.asarray(d["groups"])
    qn = list(d["qnames"])
    V = np.asarray(d["V"], float)
    V = np.where(np.isfinite(V), V, np.nanmedian(V, 0))

    # 人级聚合
    subj = np.unique(g)
    F = np.vstack([np.median(V[g == s], axis=0) for s in subj])
    Y = np.array([y[g == s][0] for s in subj])
    print(f"CM v3：{len(X)} 段 → {len(subj)} 人；SBP {Y.min():.0f}–{Y.max():.0f} "
          f"中位 {np.median(Y):.0f}，SD {Y.std():.1f}")
    print(f"特征 {F.shape[1]} 个\n")

    alphas = np.logspace(-2, 4, 25)
    n = len(Y)
    P = {k: np.zeros(n) for k in ("ridge", "pls", "single", "h1only")}
    j1 = qn.index("h1") if "h1" in qn else 0

    for i in range(n):
        tr = np.setdiff1d(np.arange(n), [i])
        m = inner_cv(F[tr], Y[tr], alphas)
        P["ridge"][i] = m["ridge"](F[[i]])[0]
        P["pls"][i] = m["pls"](F[[i]])[0]
        P["single"][i] = m["single"](F[[i]])[0]
        P["h1only"][i] = F[i, j1]
        if (i + 1) % 15 == 0:
            print(f"  LOOCV {i+1}/{n}", flush=True)

    print(f"\n{'模型':<10s}{'ρ':>8s}{'95% 区间':>18s}")
    rows = []
    for k in ("ridge", "pls", "single", "h1only"):
        r = rank(P[k], Y)
        bs = []
        for _ in range(2000):
            ii = RNG.integers(0, n, n)
            bs.append(rank(P[k][ii], Y[ii]))
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        print(f"{k:<10s}{r:>+8.3f}   [{lo:+.3f}, {hi:+.3f}]")
        rows.append(dict(model=k, rho=round(r, 4), lo=round(float(lo), 4),
                         hi=round(float(hi), 4), n=n))

    out = os.path.join(PM.DATA, "audit_cm_person_level.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
