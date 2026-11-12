"""医生标注管线预演 + 显著性门槛标定（PREVIEW，不进论文）。

目的
----
协议 §5 的主分析（L1 30 量 -> 医生盲判标签的人级判别）被逐例标签阻塞。
本脚本用 AI 预标注做**占位靶子**把管线跑通，并预先标定**显著性门槛表**：
标签置换零分布的 AUC 分位数只依赖 n 与类别构成，与标签真值无关 —— 在
医生标签到来前就能算好「AUC 多大才算超过 95% 零分位」，届时直接对照，
消除事后挑选自由度。

诚实边界（写死）：
- 占位靶子由 L1 同源量生成，跑出的 AUC/ρ 是**管线 mechanics 验证**，
  不是效应量，论文不得引用；
- 唯一论文可用的产出是门槛表（零分布分位数），因为它只依赖 n 与构成。
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
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phys_layer as PL
import probe_mediator as PM

FEATS = sorted({f for c in PL.TABLE.values() for f in c["feats"]})
N_PERM = 2000
RNG = np.random.default_rng(0)


def rank_auc(x, y_bin):
    """单量秩 AUC（Mann-Whitney，秩和法），y_bin ∈ {0,1}。"""
    x1, x0 = x[y_bin == 1], x[y_bin == 0]
    if len(x1) < 2 or len(x0) < 2:
        return np.nan
    r = pd.Series(np.concatenate([x0, x1])).rank().values
    r1 = r[len(x0):].sum()
    return float((r1 - len(x1) * (len(x1) + 1) / 2) / (len(x1) * len(x0)))


def nested_loocv_auc(X, y_bin, model="ridge"):
    """人级嵌套 LOOCV（标准化与正则在折内拟合），返回折外 AUC。

    X: (n, d)，y_bin: (n,)。n 很小（16/56），C 网格折内选。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    n = len(y_bin)
    if y_bin.min() == y_bin.max():
        return np.nan
    Cs = (0.01, 0.1, 1.0, 10.0)
    preds = np.empty(n)
    for i in range(n):
        tr = np.arange(n) != i
        sc = StandardScaler().fit(X[tr])
        # 折内选 C：留一内部再分（n 小，用内部 LOO）
        inner = np.where(tr)[0]
        best_c, best_ll = Cs[0], -np.inf
        for c in Cs:
            ll = 0.0
            for j in inner:
                it = inner != j
                if y_bin[inner[it]].min() == y_bin[inner[it]].max():
                    continue                      # 内部折只剩一类，跳过该 C 估计
                m = LogisticRegression(C=c, max_iter=2000)
                m.fit(sc.transform(X[inner[it]]), y_bin[inner[it]])
                p = m.predict_proba(sc.transform(X[j:j + 1]))[0, 1]   # j 已是原始行号
                p = min(max(p, 1e-9), 1 - 1e-9)
                ll += y_bin[j] * np.log(p) + (1 - y_bin[j]) * np.log(1 - p)
            if ll > best_ll:
                best_ll, best_c = ll, c
        m = LogisticRegression(C=best_c, max_iter=2000)
        m.fit(sc.transform(X[tr]), y_bin[tr])
        preds[i] = m.predict_proba(sc.transform(X[i:i + 1]))[0, 1]
    return rank_auc(preds, y_bin)


def main():
    a = pd.read_csv(os.path.join(PM.DATA, "ai_prelabel_cm.csv"))
    y_full = (a.prelabel == "弦").astype(int).values          # 占位靶子
    hi = (a.confidence == "高").values
    X = a[FEATS].values.astype(float)
    X = np.nan_to_num(X, nan=0.0)
    print(f"n={len(a)}  高置信子集 n={hi.sum()}（弦 {y_full[hi].sum()} / "
          f"非弦 {(1-y_full[hi]).sum()}）")

    rows = []
    # ---- 1) 门槛表：标签置换零分布的 AUC 分位数（只依赖 n 与构成） ----
    print("\n== 显著性门槛表（标签置换零分布，2000 次）==")
    print(f"{'场景':<26s}{'n':>4s}{'构成':>10s}{'AUC95':>7s}{'AUC99':>7s}")
    scenarios = {
        "高置信子集（弦 vs 非弦）": (X[hi], y_full[hi]),
        "全集（弦 vs 非弦，边界=非弦）": (X, y_full),
    }
    for name, (Xs, ys) in scenarios.items():
        n1, n0 = int(ys.sum()), int((1 - ys).sum())
        aucs = np.empty(N_PERM)
        idx1, idx0 = np.where(ys == 1)[0], np.where(ys == 0)[0]
        for t in range(N_PERM):
            perm = RNG.permutation(len(ys))
            yt = np.zeros_like(ys)
            yt[perm[: n1]] = 1
            aucs[t] = rank_auc(Xs[:, 0], yt)       # 单量列：门槛与量无关（秩统计）
        q95, q99 = np.percentile(aucs, [95, 99])
        q95 = max(q95, 1 - q95); q99 = max(q99, 1 - q99)
        print(f"{name:<26s}{len(ys):>4d}{'%d/%d' % (n1, n0):>10s}"
              f"{q95:>7.3f}{q99:>7.3f}")
        rows.append(dict(kind="门槛表", scene=name, n=len(ys), n_pos=n1,
                         auc95=round(float(q95), 3), auc99=round(float(q99), 3)))

    # ---- 2) 管线 mechanics 预演（占位靶子，结果不进论文） ----
    print("\n== 管线预演（占位靶子 = AI 预标注，PREVIEW 不进论文）==")
    for name, (Xs, ys) in scenarios.items():
        best = np.nan
        for f in FEATS:
            v = rank_auc(Xs[:, FEATS.index(f)], ys)
            if np.isfinite(v):
                v = max(v, 1 - v)
                best = v if not np.isfinite(best) or v > best else best
        auc_nested = nested_loocv_auc(Xs, ys)
        print(f"{name:<26s} 单量最优 AUC={best:.3f}  嵌套 LOOCV AUC={auc_nested:.3f}")
        rows.append(dict(kind="PREVIEW mechanics", scene=name, n=len(Xs),
                         n_pos=int(ys.sum()),
                         auc95=round(float(best), 3),
                         auc99=round(float(auc_nested), 3)))

    out = os.path.join(PM.DATA, "audit_physician_preview.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n-> {out}")
    print("注意：门槛表可用于论文（与标签真值无关）；mechanics 行仅验证管线。")


if __name__ == "__main__":
    main()
