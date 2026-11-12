"""v2.3 拆簇的簇级定位验收（G37，事前声明判据，2026-11-03）。

测什么
------
v2.3 把 reflection 拆成 reflection(p2_rel,t_notch,notch_depth,sdptg_b) 与
dicrotic(t_dicrotic,sdptg_d)。18 号已证明 v2.2 的**族级** z 只在设备轴上活，
**簇级**定位全部失败（植入 stiffness 方向 z=+0.17）。拆簇要有意义，必须证明：

    植入 dicrotic 方向 -> dicrotic 簇的 z 能定位
    植入 reflection 方向 -> reflection 簇的 z 能定位
    植入 stiffness 方向 -> 仍不可定位（阴性对照，已知）
    植入 noise 方向 -> 可定位（阳性对照，已知）
    none（SNR=0）   -> 全部簇 |z| 小（零校准）

为什么这是决定性的：dicrotic 的两列里 sdptg_d 与全部既有形态量秩相关 ≤0.25
（MIMIC/VDB 实测），若这样新信息能让簇级 z 起来，说明 v2.2 失败的根因
（簇间冗余）确实被拆掉了；若仍然平分，则拆簇在归因层无意义，只能停在
特征定义层（CONCEPT 表），如实记录。

合成数据与 audit_v22_power 相同：128 特征，前 5 个携带信号，d 集中前 5
（300:200:150:100:50），信号 = 该簇全部物理量的标准化均值方向。
基底 V 分别取 PWDB（难例：dicrotic 量与 reflection 冗余度最高）与 MIMIC
（可测域内最有利的例子）。

判据（事前声明，不许事后调整）
------------------------------
某簇"可定位" = 该簇 z 在 SNR=1 时 > +2 且随 SNR 单调，且 SNR=1 时
单次 z>+2.6 的比例 >= 60%。z 用簇标签置换零分布（nperm=200）标准化；
同时报零分布均值 mu（G39 的位置偏移在簇级可能更大）。
拆簇成功 = dicrotic 可定位 且 stiffness 不可定位。
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

NPERM = 200
REPS = 20
TARGETS = ["dicrotic", "reflection", "stiffness", "noise"]


def cluster_z(sd, G, d, seed):
    _, _, comp = sd.compose(G, d)
    fc, _ = PM.attr_null(sd, G, d, nperm=NPERM, seed=seed)
    mu, sg = fc.mean(0), fc.std(0)
    z = (comp - mu) / np.where(sg > 1e-12, sg, 1.0)
    return comp, z, mu


def main():
    rows = []
    for src in ("PWDB", "MIMIC"):
        d0 = PM.load_source(src)
        V = d0["V"].astype(np.float64)
        V = np.where(np.isfinite(V), V, np.nanmedian(V, 0))
        qn = list(d0["qnames"])
        n, p = V.shape
        idx_of = {q: j for j, q in enumerate(qn)}
        Z = (V - V.mean(0)) / (V.std(0) + 1e-12)
        print(f"\n[{src}] n={n} p={p} 簇={len(PL.TABLE)} hash={PL.check_frozen()}")

        dirs = {}
        for cl in TARGETS:
            cols = [idx_of[q] for q in PL.TABLE[cl]["feats"] if q in idx_of]
            u = np.zeros(p)
            u[cols] = 1.0
            dirs[cl] = u / np.linalg.norm(u)

        for tag in TARGETS + ["none"]:
            sig = None if tag == "none" else Z @ dirs[tag]
            for snr in (0.0, 0.5, 1.0, 2.0):
                if tag == "none" and snr > 0:
                    continue
                zs, mus = [], []
                sd = None
                for rep in range(REPS):
                    rng = np.random.default_rng(1000 * int(snr * 2) + rep +
                                                (0 if tag == "none" else hash(tag) % 977))
                    A = rng.normal(size=(n, 128))
                    if tag != "none":
                        for j in range(5):
                            A[:, j] = snr * sig + rng.normal(size=n)
                    if sd is None or tag == "none":
                        sd = PM.ShapleyAttrib(V, qn)
                    Az = (A - A.mean(0)) / (A.std(0) + 1e-12)
                    G = sd.Z.T @ Az / n
                    comp, z, mu = cluster_z(sd, G, np.array(
                        [300., 200., 150., 100., 50.] + [1.] * 123), int(rng.integers(1e6)))
                    zs.append(z)
                    mus.append(mu)
                zs = np.array(zs)
                zi = {c: i for i, c in enumerate(sd.cs)}
                for cl in ("dicrotic", "reflection", "stiffness", "noise"):
                    k = zi[cl]
                    rows.append(dict(
                        source=src, direction=tag, snr=snr, cluster=cl,
                        comp=round(float(zs[:, k].mean()), 4),
                        z=round(float(zs[:, k].mean()), 3),
                        z_sd=round(float(zs[:, k].std()), 3),
                        null_mu=round(float(np.mean(mus, axis=0)[k]), 4),
                        det=round(float((zs[:, k] > 2.6).mean()), 3)))
                k = zi[tag] if tag != "none" else None
                if k is not None:
                    print(f"  {tag:<10s} SNR={snr:<3} 植入簇 comp={zs[:, k].mean():+.3f} "
                          f"z={zs[:, k].mean():+.2f}±{zs[:, k].std():.2f} "
                          f"mu={np.mean(mus, axis=0)[k]:+.3f} det(z>2.6)={100*(zs[:, k]>2.6).mean():.0f}%")

    fn = os.path.join(PM.DATA, "audit_v23_split.csv")
    with open(fn, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {fn} ({len(rows)} rows)")

    print("\n判读（判据：植入簇 z 在 SNR=1 时 >+2 且单调，det>=60% -> 可定位）")
    for src in ("PWDB", "MIMIC"):
        for tag in TARGETS:
            g = {r["snr"]: r for r in rows
                 if r["source"] == src and r["direction"] == tag and r["cluster"] == tag}
            if not g:
                continue
            zs = [g[s]["z"] for s in (0.0, 0.5, 1.0, 2.0)]
            det = g[1.0]["det"]
            loc = zs[2] > 2 and zs[3] >= zs[2] >= zs[1] and det >= 0.6
            print(f"  {src:<6s} {tag:<10s} z: {zs[0]:+.2f} {zs[1]:+.2f} {zs[2]:+.2f} {zs[3]:+.2f}"
                  f"  det(SNR=1)={100*det:.0f}%  -> {'可定位' if loc else '不可定位'}")


if __name__ == "__main__":
    main()
