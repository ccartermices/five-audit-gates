"""份额估计对加权口径的稳健性：主导特征是否在制造"什么都测不到"。

为什么必须做
------------
`audit_null_family` 显示 d_f = |c_f * mean(a_f)| 被 3-5 个特征拿走 71-88%，
零分布 SD 因此高达 0.10-0.18，与待检验效应同量级 —— 没有功效。
但那可能只是"加权方式把少数特征放大"造成的，而不是数据里没有信号。
换两种加权重做：

  mean   现行口径，d_f = |c_f * mean(a_f)|      （12 号）
  equal  等权，d_f = 1                          （每个存活特征一票）
  sd     按贡献波动，d_f = |c_f * sd(a_f)|       （对预测有影响的量）

若三种口径都给出"与偶然无差"，结论就不是加权造成的。若等权下信号出现，
说明信号存在但集中在少数特征上，报告口径应当改。
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
NPERM = 100


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

    idx_of = {q: j for j, q in enumerate(names)}
    groups = [(c, [idx_of[q] for q in v["feats"]]) for c, v in PL.TABLE.items()]
    fams = [PL.family_of(c) for c, _ in groups]
    st = np.load(os.path.join(DATA, "probe_mediator.npz"), allow_pickle=True)
    out = []
    print(f"{'源':<7s}{'架构':<13s}{'加权':<7s}{'族':<12s}"
          f"{'实测':>8s}{'零均值':>8s}{'零SD':>7s}{'z':>7s}{'P(零>=)':>9s}")
    for s in SRC:
        P = np.column_stack([pc1(VI[s], idx) for _, idx in groups])
        for a in ARCH:
            for wmode in ("mean", "equal", "sd"):
                zs = {f: [] for f in fams}
                obs = {f: [] for f in fams}
                nn = []
                for sd in (0, 1, 2):
                    k = f"{s}_{a}_s{sd}"
                    A, c = st[k + "_A"], st[k + "_c"]
                    rzA = _rz(A)
                    am, asd = A.mean(0), A.std(0)
                    if wmode == "mean":
                        d = np.abs(c * am)
                    elif wmode == "equal":
                        d = np.ones_like(am)
                    else:
                        d = np.abs(c * asd)
                    rhoC = (rzA.T @ _rz(P)) / len(P)
                    alive = (np.abs(np.nan_to_num(rhoC)).max(1) > 1e-12) & (am > 0)
                    o = fam_share(rhoC, d, alive, fams)
                    for f in fams:
                        obs[f].append(o[f])
                    null = {f: [] for f in fams}
                    for p in range(NPERM):
                        Pp = P[np.random.default_rng(7000 + p).permutation(len(P))]
                        rp = (rzA.T @ _rz(Pp)) / len(Pp)
                        sh = fam_share(rp, d, alive, fams)
                        for f in fams:
                            null[f].append(sh[f])
                    for f in fams:
                        v = np.array(null[f])
                        zs[f].append((o[f] - v.mean()) / (v.std() + 1e-12))
                    nn.append(np.mean(v.std() * 0 + np.std([x for x in null[fams[0]]])))
                for f in fams:
                    print(f"{s:<7s}{a:<13s}{wmode:<7s}{f:<12s}"
                          f"{np.mean(obs[f]):>8.3f}{'-':>8s}{'-':>7s}"
                          f"{np.mean(zs[f]):>+7.2f}{'-':>9s}")
                    out.append(dict(source=s, arch=a, weighting=wmode, family=f,
                                    observed=round(float(np.mean(obs[f])), 4),
                                    z=round(float(np.mean(zs[f])), 3)))
    with open(os.path.join(DATA, "audit_weighting.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    print("\n-> data/audit_weighting.csv")


if __name__ == "__main__":
    main()
