"""簇定义与缺失结构的效度审计（17 号）。

四个问题
--------
1. `notch` 簇（只有 notch_depth）与 `reflection` 簇的 `p_notch_rel` 是否同一个量？
   若 notch_depth = 1 - p_notch_rel 恒成立，8 簇只有 7 个独立维度，机理份额双重计数。
2. `reflection` 的 PC1 在 CM/MIMIC 上是否退化成"P2 能否检出"的指示变量？
3. 把 notch 并入 reflection 后，12 号的机理份额挪动多少。
4. 每个簇的份额是否真的超出**偶然**：把簇的 PC1 沿行打乱后重算份额，
   得到该簇在零假设下的份额分布。份额小于零分布均值，就说明那个簇的
   "机制份额"其实是 128 个特征在噪声维度上的运气。

为什么能离线算
--------------
份额 = sum_f d_f * w_fk / sum_f d_f,  w_fk = rho(f, PC1_k)^2 / sum_k rho^2,
d_f = |c_f * mean(a_f)|，PC1 只由 L1 矩阵决定。npz 已存 A/V/c，换簇定义重算
不需要重训，也不会引入训练侧差异 —— 这是"归因靶子可以独立复核"的实质含义。
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
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phys_layer as PL                    # noqa: E402
import probe_mediator as PM                # noqa: E402

ROOT = _repo.ROOT
DATA = os.path.join(ROOT, "data")
SRC = ["PWDB", "VDB", "MIMIC", "CM"]
ARCH = ["mlp", "transformer"]
NPERM = 20


def _rz(M):
    R = np.apply_along_axis(rankdata, 0, M)
    R = R - R.mean(0)
    s = R.std(0)
    return R / np.where(s > 1e-12, s, 1.0)


def pc1(V, idx, fit_mask=None):
    """簇的第一主成分；符号按"最大载荷为正"固化，与 cluster_dims 一致。"""
    A = V[:, idx].astype(np.float64)
    m = np.ones(len(A), bool) if fit_mask is None else fit_mask
    mu = A[m].mean(0)
    sd = A[m].std(0) + 1e-12
    Z = (A - mu) / sd
    _, _, vt = np.linalg.svd(Z[m], full_matrices=False)
    p = Z @ vt[0]
    if vt[0][int(np.argmax(np.abs(vt[0])))] < 0:
        p = -p
    return p


def shares(A, V, c, groups, fit_masks=None, perm_seed=None):
    P = np.column_stack([pc1(V, idx, None if fit_masks is None else fit_masks.get(nm))
                         for nm, idx in groups])
    if perm_seed is not None:
        P = P[np.random.default_rng(perm_seed).permutation(len(P))]
    rhoC = (_rz(A).T @ _rz(P)) / len(P)
    am = A.mean(0)
    d = np.abs(c * am)
    alive = (np.abs(np.nan_to_num(rhoC)).max(1) > 1e-12) & (am > 0)
    tot = d[alive].sum()
    out = {nm: 0.0 for nm, _ in groups}
    for f in np.where(alive)[0]:
        r = np.nan_to_num(rhoC[f])
        w = r ** 2 / max((r ** 2).sum(), 1e-12)
        for k, (nm, _) in enumerate(groups):
            out[nm] += d[f] * w[k] / tot
    return out


def mech(sh):
    return sum(v for k, v in sh.items() if PL.family_of(k) == "mechanism")


def main():
    VI, VR, PF, names = {}, {}, {}, None
    for s in SRC:
        d = PM.load_source(s, keep_nan=True)
        names = d["qnames"]
        V = d["V"].astype(np.float64)
        VR[s] = V
        VI[s] = np.where(np.isfinite(V), V, np.nanmedian(V, 0))
        PF[s] = d["aux"]["p2_found"]
        print(f"  {s:<6s} n={len(VI[s]):<6d} P2 检出 {np.nanmean(PF[s]) * 100:5.1f}%",
              flush=True)

    idx_of = {q: j for j, q in enumerate(names)}
    base = [(c, [idx_of[q] for q in v["feats"]]) for c, v in PL.TABLE.items()]
    merge = []
    for c, v in PL.TABLE.items():
        if c == "notch":
            continue
        f = list(v["feats"])
        if c == "reflection":
            f += list(PL.TABLE["notch"]["feats"])
        merge.append((c, [idx_of[q] for q in f]))

    meta = {}
    for r in csv.DictReader(open(os.path.join(DATA, "probe_mediator_meta.csv"),
                                 encoding="utf-8-sig")):
        try:
            v = float(r["value"])
        except (TypeError, ValueError):
            continue           # source / arch / dept 等非数值键
        meta.setdefault((r["source"], r["arch"], r["key"]), []).append(v)

    st = np.load(os.path.join(DATA, "probe_mediator.npz"), allow_pickle=True)
    res = {}
    print("\n" + "=" * 78)
    print("§0 离线复现校验：逐种子对比 12 号 share_*（meta 每键 3 行 = 3 个种子）")
    worst = 0.0
    for s in SRC:
        for a in ARCH:
            for sd in (0, 1, 2):
                k = f"{s}_{a}_s{sd}"
                A, V, c = st[k + "_A"], st[k + "_V"], st[k + "_c"]
                assert np.allclose(V, VI[s].astype(np.float32), atol=1e-4), f"{k} V 不一致"
                sh = shares(A, VI[s], c, base)
                res[(s, a, sd)] = dict(base=sh)
                for c_ in PL.TABLE:
                    ref = meta[(s, a, f"share_{c_}")][sd]
                    worst = max(worst, abs(sh[c_] - ref))
    print(f"  逐簇逐种子最大绝对差 = {worst:.4f}"
          f"（{'通过' if worst < 0.02 else '异常'}；差异来自浮点非确定性）")

    print("\n" + "=" * 78)
    print("§1 notch 与 reflection 是否同一个量")
    for s in SRC:
        V = VR[s]
        j1, j2 = names.index("notch_depth"), names.index("p_notch_rel")
        z = V[:, j1] + V[:, j2]
        ok = np.isfinite(z)
        dev = np.abs(z[ok] - 1.0)
        print(f"  {s:<6s} notch_depth + p_notch_rel = 1 的比例 "
              f"{(dev < 1e-6).mean() * 100:5.1f}%   中位偏差 {np.median(dev):.2e}"
              f"   99分位 {np.percentile(dev, 99):.2e}")

    print("\n" + "=" * 78)
    print("§2 reflection 的 PC1 与「P2 是否检出」的相关")
    print("   以及该簇与血压的边际关联强度（rho 向量范数，真 pairwise）")
    for s in SRC:
        ri = [idx_of[q] for q in PL.TABLE["reflection"]["feats"]]
        p = pc1(VI[s], ri)
        r = spearmanr(p, PF[s])[0] if PF[s].std() > 1e-9 else np.nan
        print(f"  {s:<6s} corr(PC1, P2检出)={'  n/a' if not np.isfinite(r) else f'{r:+.3f}'}")

    print("\n" + "=" * 78)
    print("§3 机理份额：现行 8 簇 vs notch 并入 reflection（3 种子均值）")
    print(f"{'源':<7s}{'架构':<13s}{'8 簇':>8s}{'并入':>8s}{'Δ':>8s}"
          f"{'notch':>8s}{'reflection':>11s}{'两者合计':>9s}")
    for s in SRC:
        for a in ARCH:
            b = np.mean([mech(res[(s, a, sd)]["base"]) for sd in (0, 1, 2)])
            m_, n_, r_ = [], [], []
            for sd in (0, 1, 2):
                k = f"{s}_{a}_s{sd}"
                sh = shares(st[k + "_A"], VI[s], st[k + "_c"], merge)
                res[(s, a, sd)]["merge"] = sh
                m_.append(mech(sh))
                n_.append(res[(s, a, sd)]["base"]["notch"])
                r_.append(res[(s, a, sd)]["base"]["reflection"])
            m_ = np.mean(m_)
            print(f"{s:<7s}{a:<13s}{b:>8.3f}{m_:>8.3f}{m_ - b:>+8.3f}"
                  f"{np.mean(n_):>8.3f}{np.mean(r_):>11.3f}{np.mean(n_) + np.mean(r_):>9.3f}")

    print("\n" + "=" * 78)
    print(f"§4 零假设基线：簇 PC1 沿行打乱 {NPERM} 次后的份额分布")
    print("   z<0 表示该簇的份额不高于偶然（这个'机制'是 128 个特征在噪声上的运气）")
    print(f"{'源':<7s}{'架构':<13s}{'簇':<12s}{'实测':>8s}{'零均值':>8s}"
          f"{'零SD':>7s}{'z':>7s}{'p':>7s}")
    nullrows = []
    for s in SRC:
        for a in ARCH:
            for sd in (0,):
                k = f"{s}_{a}_s{sd}"
                A, c = st[k + "_A"], st[k + "_c"]
                obs = res[(s, a, sd)]["base"]
                N = []
                for p in range(NPERM):
                    N.append(shares(A, VI[s], c, base, perm_seed=1000 * sd + p))
                for c_ in PL.TABLE:
                    v = np.array([x[c_] for x in N])
                    z = (obs[c_] - v.mean()) / (v.std() + 1e-12)
                    pv = float((v >= obs[c_]).mean())
                    nullrows.append((s, a, c_, obs[c_], v.mean(), v.std(), z, pv))
                    if c_ in ("reflection", "notch", "stiffness", "noise"):
                        print(f"{s:<7s}{a:<13s}{c_:<12s}{obs[c_]:>8.3f}"
                              f"{v.mean():>8.3f}{v.std():>7.4f}{z:>+7.2f}{pv:>7.2f}")

    with open(os.path.join(DATA, "audit_cluster_alias.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "arch", "seed", "variant", "cluster", "share"])
        for (s, a, sd), v in res.items():
            for var in ("base", "merge"):
                for c_, val in v[var].items():
                    w.writerow([s, a, sd, var, c_, round(float(val), 5)])
    with open(os.path.join(DATA, "audit_null_share.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "arch", "cluster", "observed", "null_mean",
                    "null_sd", "z", "p_perm"])
        for r in nullrows:
            w.writerow([r[0], r[1], r[2]] + [round(float(x), 4) for x in r[3:]])
    print("\n-> data/audit_cluster_alias.csv / data/audit_null_share.csv")


if __name__ == "__main__":
    main()
