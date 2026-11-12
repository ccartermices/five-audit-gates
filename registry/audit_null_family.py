"""三族份额的偶然基线：12 号报的 50-64% 究竟离随机有多远。

问题
----
L1 的 8 个簇里，5 个属 mechanism、2 个属 descriptor、1 个属 device。若每个特征对各簇
的归因在零假设下是均匀的，三族份额的期望就是 5/8=62.5% / 2/8=25% / 1/8=12.5%。
12 号报"机理占 50-64%"，其中 PWDB 64.1% 与 62.5% 只差 1.6 个百分点 —— 这个数
不比较基线就没有意义。

更麻烦的是方差。份额 = sum_f d_f*w_f / sum_f d_f，若 d_f 被少数特征主导，
份额就不是 128 个数的平均，而是几个数的抽取，零分布会非常宽。因此必须给出
（a）d_f 的集中度；（b）按行打乱后的三族份额分布；（c）实测值在该分布中的位置。

为什么用行打乱做零假设
--------------------
打乱簇 PC1 的行序，等于保留"簇内测量结构"与"特征激活分布"，只切断
"特征 -> 生理量"的对应。这正是要检验的那条通路。
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
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phys_layer as PL                    # noqa: E402
import probe_mediator as PM                # noqa: E402

ROOT = _repo.ROOT
DATA = os.path.join(ROOT, "data")
SRC = ["PWDB", "VDB", "MIMIC", "CM"]
ARCH = ["mlp", "transformer"]
NPERM = 200
SEEDS = (0, 1, 2)


def _rz(M):
    R = np.apply_along_axis(rankdata, 0, M)
    R = R - R.mean(0)
    s = R.std(0)
    return R / np.where(s > 1e-12, s, 1.0)


def pc1(V, idx):
    A = V[:, idx].astype(np.float64)
    Z = (A - A.mean(0)) / (A.std(0) + 1e-12)
    _, _, vt = np.linalg.svd(Z, full_matrices=False)
    p = Z @ vt[0]
    if vt[0][int(np.argmax(np.abs(vt[0])))] < 0:
        p = -p
    return p


def fam_share(rhoC, d, alive, fams):
    tot = d[alive].sum()
    out = {f: 0.0 for f in fams}
    for f in np.where(alive)[0]:
        r = np.nan_to_num(rhoC[f])
        w = r ** 2 / max((r ** 2).sum(), 1e-12)
        for k, fm in enumerate(fams):
            out[fm] += d[f] * w[k] / tot
    return out


def main():
    VI, names = {}, None
    for s in SRC:
        d = PM.load_source(s, keep_nan=True)
        names = d["qnames"]
        V = d["V"].astype(np.float64)
        VI[s] = np.where(np.isfinite(V), V, np.nanmedian(V, 0))
        print(f"  {s:<6s} n={len(VI[s])}", flush=True)

    idx_of = {q: j for j, q in enumerate(names)}
    groups = [(c, [idx_of[q] for q in v["feats"]]) for c, v in PL.TABLE.items()]
    fams = [PL.family_of(c) for c, _ in groups]
    print(f"\n簇 -> 族：{dict(zip([c for c, _ in groups], fams))}")
    cnt = {f: fams.count(f) for f in set(fams)}
    base = {f: cnt[f] / len(fams) for f in cnt}
    print("随机分配下的期望：" + "  ".join(f"{k} {v * 100:.1f}%" for k, v in base.items()))

    meta = {}
    for r in csv.DictReader(open(os.path.join(DATA, "probe_mediator_meta.csv"),
                                 encoding="utf-8-sig")):
        try:
            v = float(r["value"])
        except (TypeError, ValueError):
            continue
        meta.setdefault((r["source"], r["arch"], r["key"]), []).append(v)

    st = np.load(os.path.join(DATA, "probe_mediator.npz"), allow_pickle=True)
    out = []
    for s in SRC:
        V = VI[s]
        P = np.column_stack([pc1(V, idx) for _, idx in groups])
        for a in ARCH:
          for sd in SEEDS:
            k = f"{s}_{a}_s{sd}"
            A, c = st[k + "_A"], st[k + "_c"]
            rzA = _rz(A)
            am = A.mean(0)
            d = np.abs(c * am)
            rhoC = (rzA.T @ _rz(P)) / len(P)
            # 存活掩码由实测确定并固定：若每个置换各自重算，tot 会随置换漂移，
            # 零分布就不只是"切断特征-生理对应"，还混入了分母变化。
            alive = (np.abs(np.nan_to_num(rhoC)).max(1) > 1e-12) & (am > 0)
            obs = fam_share(rhoC, d, alive, fams)

            # d_f 集中度：解释力被几个特征拿走
            dd = np.sort(d[alive])[::-1]
            top1 = dd[0] / dd.sum()
            top5 = dd[:5].sum() / dd.sum()
            n50 = int(np.searchsorted(np.cumsum(dd) / dd.sum(), 0.5) + 1)

            # 零分布
            null = {f: [] for f in fams}
            for p in range(NPERM):
                Pp = P[np.random.default_rng(90000 + p).permutation(len(P))]
                rp = (rzA.T @ _rz(Pp)) / len(Pp)
                sh = fam_share(rp, d, alive, fams)
                for f in fams:
                    null[f].append(sh[f])

            print(f"\n{s}/{a}/s{sd}  d 集中度：top1 {top1 * 100:.0f}%  top5 {top5 * 100:.0f}%  "
                  f"到 50% 需要 {n50} 个特征（共 {alive.sum()} 个存活）")
            print(f"  {'族':<12s}{'实测':>8s}{'零均值':>8s}{'零SD':>7s}"
                  f"{'基线':>7s}{'z':>7s}{'P(零>=实测)':>12s}")
            for f in ["mechanism", "descriptor", "device"]:
                v = np.array(null[f])
                z = (obs[f] - v.mean()) / (v.std() + 1e-12)
                pv = float((v >= obs[f]).mean())
                print(f"  {f:<12s}{obs[f]:>8.3f}{v.mean():>8.3f}{v.std():>7.3f}"
                      f"{base[f]:>7.3f}{z:>+7.2f}{pv:>12.3f}")
                out.append(dict(source=s, arch=a, seed=sd, family=f,
                                observed=round(obs[f], 4),
                                null_mean=round(float(v.mean()), 4),
                                null_sd=round(float(v.std()), 4),
                                baseline=round(base[f], 4), z=round(float(z), 3),
                                p_perm=round(pv, 3),
                                d_top1=round(float(top1), 4),
                                d_top5=round(float(top5), 4), n50=n50))


    with open(os.path.join(DATA, "audit_null_family.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    print("\n-> data/audit_null_family.csv")


if __name__ == "__main__":
    main()
