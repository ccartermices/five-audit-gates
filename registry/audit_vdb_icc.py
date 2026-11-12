"""VDB 搏间信度（ICC）：L1 v2.3 全部 30 量，多人多搏重抽后可测。

动机
----
规范 GAPS 明列「VDB 一人一段无法测信度」。fetch_vitaldb_multibeat.py
已重抽 98 例 x 30 搏 = 2940 搏（主导心律过滤保留，cons 门**未**施加，
cons 范围 0.188–1.0，搏间变异是研究对象）。

口径
----
- ICC(1,1)：单搏可用性（随机效应，单次测量）。
- ICC(1,k)：k 搬平均后的信度（k=30，即"用 30 搏中位数"时）。
  k 大时 ICC(1,k) ≈ BMS/(BMS+WMS) 的上界化形式，恒 >= ICC(1,1)。
- 判据（Koo & Li 2016）：>=0.75 好、0.50–0.75 中、<0.50 差。
- 对照：MIMIC 425 人 x 30 段做同口径段间 ICC —— 段间与搏间不是同一
  口径，但对照能回答"搏级噪声比段级噪声多多少"。

注意：CM/发搏数据的 ICC(1,1) 不可与此直接比（ep9 是段间重复采集，
含重测间隔）；表内只放 VDB 搏间与 MIMIC 段间两列。
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
from pwfeat import seg_feats

FEATS = sorted({f for c in PL.TABLE.values() for f in c["feats"]})


def sdptg_cols(X, idx=None):
    from pwfeat import sdptg_waves
    W = [sdptg_waves(X[i], 9) for i in (idx if idx is not None else range(len(X)))]
    return (np.array([w["b"] for w in W]), np.array([w["d"] for w in W]))


def icc_oneway(x, g):
    """ICC(1,1) 与 ICC(1,k)，不平衡允许。x: 量测, g: 组 id。"""
    df = pd.DataFrame({"x": x, "g": g}).dropna()
    k = df.g.nunique()
    n = len(df)
    if k < 2 or n <= k:
        return np.nan, np.nan
    grand = df.x.mean()
    gm = df.groupby("g").x.agg(["mean", "size", "var"])
    k0 = n / k
    ssb = float((gm["size"] * (gm["mean"] - grand) ** 2).sum())
    ssw = float(((df.x - df.g.map(gm["mean"])) ** 2).sum())
    msb = ssb / (k - 1)
    msw = ssw / (n - k)
    if msb + (k0 - 1) * msw <= 0:
        return np.nan, np.nan
    icc1 = (msb - msw) / (msb + (k0 - 1) * msw)
    icck = (msb - msw) / msb if msb > 0 else np.nan
    return float(icc1), float(icck)


def band(x):
    return ("好" if x >= 0.75 else "中" if x >= 0.50 else "差") if np.isfinite(x) else "-"


def main():
    rows = []

    # ---------- VDB 多搏 ----------
    vd = os.path.join(PM.DATA, "vitaldb")
    z = np.load(os.path.join(vd, "multibeat_cyc.npz"))
    m = pd.read_csv(os.path.join(vd, "multibeat_meta.csv"))
    # X/caseid/beat 只含 ok==1 的搏（fetch 脚本 append 顺序即此顺序）；
    # meta 还含失败 case 行，不能用 meta 过滤 X（长度不一致，2940 vs 2972）。
    X = z["X"]
    caseid = z["caseid"]
    assert len(X) == len(caseid) == (m.ok == 1).sum(), "multibeat X/meta 不对齐"
    print(f"VDB 多搏：{X.shape[0]} 搏 / {pd.unique(caseid).size} 例", flush=True)
    F = pd.DataFrame([seg_feats(x) for x in X])
    g = pd.factorize(caseid)[0]
    for f in FEATS:
        i1, ik = icc_oneway(F[f].values, g)
        rows.append(dict(source="VDB_搏间", feat=f, n_person=int(pd.unique(caseid).size),
                         k_median=30, icc1=round(i1, 4) if np.isfinite(i1) else "",
                         icck=round(ik, 4) if np.isfinite(ik) else "", band=band(i1)))
        print(f"VDB  {f:<12s} ICC(1,1)={i1:+.3f}  ICC(1,k)={ik:+.3f}  {band(i1)}")

    # ---------- MIMIC 段间对照（features 表现成值；sdptg 表里没有，现算） ----------
    Fm = pd.read_csv(os.path.join(PM.DATA, "mimicbp_features.csv"))
    zm = np.load(os.path.join(PM.DATA, "mimicbp_cycles.npz"))
    assert len(zm["X"]) == len(Fm), "MIMIC cycles/features 不对齐"
    sbm, sdm = sdptg_cols(zm["X"].astype(float))
    Fm["sdptg_b"] = sbm
    Fm["sdptg_d"] = sdm
    g_pid = pd.factorize(Fm.pid)[0]
    print(flush=True)
    for f in FEATS:
        if f not in Fm.columns:
            rows.append(dict(source="MIMIC_段间", feat=f, n_person=425, k_median=30,
                             icc1="", icck="", band="-"))
            continue
        i1, ik = icc_oneway(Fm[f].values, g_pid)
        rows.append(dict(source="MIMIC_段间", feat=f, n_person=425,
                         k_median=30, icc1=round(i1, 4) if np.isfinite(i1) else "",
                         icck=round(ik, 4) if np.isfinite(ik) else "", band=band(i1)))
        print(f"MIMIC {f:<12s} ICC(1,1)={i1:+.3f}  ICC(1,k)={ik:+.3f}  {band(i1)}")

    out = os.path.join(PM.DATA, "audit_vdb_icc.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n-> {out}")

    # 汇总：两簇关键量
    print("\n== dicrotic / reflection 关键量 ==")
    for f in ("t_dicrotic", "sdptg_d", "p2_rel", "t_notch", "notch_depth", "sdptg_b"):
        v = [r for r in rows if r["feat"] == f]
        s = "  ".join(f"{r['source']} ICC1={r['icc1']}" for r in v)
        print(f"{f:<12s} {s}")


if __name__ == "__main__":
    main()
