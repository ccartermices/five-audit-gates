"""v2.2 归因框架的验收对照（规范 G37 要求留档）。

测什么
------
在同一条合成数据上并排跑两代归因口径，唯一的差别是靶子本身：

  旧 v2.1   8 个簇的 PC1，rho^2 在 8 个相关方向间分摊
  新 v2.2   按簇做 Shapley R^2 分解（无基底，精确可加）

合成数据：128 个特征中只有前 5 个携带信号，d 的集中度按实测设置（top5 占约 80%）。
信号取自一个 L1 方向的单变量：

  stiffness / shape / spectrum / noise   该簇全部物理量的标准化均值方向
  none（SNR=0）                          与 L1 无关的纯噪声 —— 这才是真阴性对照

为什么"随机方向"不能当阴性对照
------------------------------
29 维里随机一个方向仍然是真实的 L1 线性组合，被检出是对的，不是误报。
第一版把随机方向当阴性，得到 z=+4 就误判成"误报率 3/4"，那是我的对照设计错。

判据（事前声明，2026-11-03）
---------------------------
某族的 z 在 SNR=1 时 > +2，且随 SNR 单调上升 -> 该族"可定位"
SNR=0 时三族 |z| < 2                              -> 零分布校准正常
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
import phys_layer as PL          # noqa: E402
import probe_mediator as PM      # noqa: E402

DATA = PM.DATA
NPERM = 200
FAMS = ("mechanism", "descriptor", "device")
TARGETS = ["stiffness", "shape", "spectrum", "noise"]


def rz(M):
    from scipy.stats import rankdata
    R = np.apply_along_axis(rankdata, 0, M)
    R = R - R.mean(0)
    s = R.std(0)
    return R / np.where(s > 1e-12, s, 1.0)


def old_shares(rzA, rzP, lsq, d, alive, tot):
    rho = (rzA.T @ rzP) / len(rzA)
    rho[~alive] = 0.0
    sq = np.nan_to_num(rho) ** 2
    W = sq / np.maximum(sq.sum(1, keepdims=True), 1e-12)
    return (d[alive] @ (W[alive] @ lsq)) / tot


def fam_vec(comp, cs):
    out = []
    for f in FAMS:
        out.append(sum(comp[i] for i, c in enumerate(cs) if PL.family_of(c) == f))
    return np.array(out)


def run_new(A, V, qn, d, rng):
    sd = PM.ShapleyAttrib(V, qn)
    Az = (A - A.mean(0)) / (A.std(0) + 1e-12)
    G = sd.Z.T @ Az / len(A)
    _, cov, comp = sd.compose(G, d)
    obs = fam_vec(comp, sd.cs)
    fc, ff = PM.attr_null(sd, G, d, nperm=NPERM, seed=int(rng.integers(1e6)))
    z = (obs - ff.mean(0)) / (ff.std(0) + 1e-12)
    return obs, z, cov, None


def run_old(A, V, qn, d, rng):
    P, cs = PM.cluster_dims(V, qn)
    lsq = np.eye(P.shape[1])
    rzA, rzP = rz(A), rz(P)
    alive = np.abs(np.nan_to_num((rzA.T @ rzP) / len(A))).max(1) > 1e-12
    tot = d[alive].sum() + 1e-12
    obs_c = old_shares(rzA, rzP, lsq, d, alive, tot)
    null = np.array([old_shares(rzA, rzP[rng.permutation(len(P))], lsq, d, alive, tot)
                     for _ in range(NPERM)])
    obs = fam_vec(obs_c, cs)
    nl = np.array([fam_vec(x, cs) for x in null])
    z = (obs - nl.mean(0)) / (nl.std(0) + 1e-12)
    return obs, z, float(np.nansum(d) / np.nansum(d)), None


def main():
    d0 = PM.load_source("PWDB")
    V = d0["V"].astype(np.float64)
    V = np.where(np.isfinite(V), V, np.nanmedian(V, 0))
    qn = list(d0["qnames"])
    n, p = V.shape
    idx_of = {q: j for j, q in enumerate(qn)}
    Z = (V - V.mean(0)) / (V.std(0) + 1e-12)
    print(f"[PWDB] n={n} p={p} 簇={len(PL.TABLE)} hash={PL.check_frozen()}")

    dirs = {}
    for cl in TARGETS:
        cols = [idx_of[q] for q in PL.TABLE[cl]["feats"] if q in idx_of]
        u = np.zeros(p); u[cols] = 1.0
        dirs[cl] = u / np.linalg.norm(u)

    rows = []
    print(f"\n{'方向':<11s}{'SNR':>5s}{'口径':<7s}{'覆盖%':>7s}"
          f"{'机理%':>7s}{'z':>7s}{'重述%':>7s}{'z':>7s}{'噪声%':>7s}{'z':>7s}")
    for tag in TARGETS + ["none"]:
        sig = None if tag == "none" else Z @ dirs[tag]
        for snr in (0.0, 0.5, 1.0, 2.0):
            if tag == "none" and snr > 0:
                continue
            rng = np.random.default_rng(900 + int(snr * 10))
            A = rng.normal(size=(n, 128))
            if tag != "none":
                for j in range(5):
                    A[:, j] = snr * sig + rng.normal(size=n)
            d = np.ones(128); d[:5] = [300, 200, 150, 100, 50]
            for meth, fn in (("v2.1", run_old), ("v2.2", run_new)):
                obs, z, cov, _ = fn(A, V, qn, d, rng)
                lab = "none" if tag == "none" else tag
                print(f"{lab:<11s}{snr:>5.1f}{meth:<7s}{100*cov:>7.1f}"
                      f"{100*obs[0]:>7.1f}{z[0]:>+7.2f}"
                      f"{100*obs[1]:>7.1f}{z[1]:>+7.2f}"
                      f"{100*obs[2]:>7.1f}{z[2]:>+7.2f}")
                rows.append(dict(direction=lab, snr=snr, method=meth,
                                 coverage=round(cov, 4),
                                 mech=round(obs[0], 4), mech_z=round(z[0], 3),
                                 desc=round(obs[1], 4), desc_z=round(z[1], 3),
                                 device=round(obs[2], 4), device_z=round(z[2], 3)))
            print()

    fn = os.path.join(DATA, "audit_v22_power.csv")
    with open(fn, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    print("=" * 84)
    for meth in ("v2.1", "v2.2"):
        sub = [r for r in rows if r["method"] == meth]
        print(f"\n{meth} 判读（判据：目标族 z 在 SNR=1 时 >+2 且随 SNR 单调 -> 可定位）")
        for tag in TARGETS:
            g = [r for r in sub if r["direction"] == tag]
            key = {"stiffness": "mech_z", "shape": "desc_z",
                   "spectrum": "desc_z", "noise": "device_z"}[tag]
            zs = [r[key] for r in g]
            mono = all(zs[i] <= zs[i + 1] + 0.2 for i in range(len(zs) - 1))
            ok = zs[2] > 2 and mono
            print(f"  {tag:<10s} z(S=0,0.5,1,2)={['%+.2f' % v for v in zs]} "
                  f"单调~{mono}  {'可定位' if ok else '不可定位'}")
        ng = [r for r in sub if r["direction"] == "none"]
        if ng:
            r = ng[0]
            print(f"  零校准（SNR=0）：z=({r['mech_z']:+.2f}, {r['desc_z']:+.2f}, "
                  f"{r['device_z']:+.2f})  {'正常' if max(abs(r['mech_z']), abs(r['desc_z']), abs(r['device_z'])) < 2 else '偏移'}")
    print(f"\n-> {fn}")


if __name__ == "__main__":
    main()
