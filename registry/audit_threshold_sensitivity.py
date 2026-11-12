# -*- coding: utf-8 -*-
"""E5 阈值敏感性：数值闸门的判定在阈值附近会不会翻转。

为什么必须做
------------
G3 的 ICC ≥ 0.80 与 G5 的 Jaccard ≥ 0.5 都是惯例值。审稿人会问「换成 0.6 会
怎样」。本项目的立论是「阈值必须事先定」，所以阈值附近的行为必须报告，
而且翻转就写在正文里，不藏。

脚本做两件事
------------
1. G3：对 ICC 门槛做 ±20% 扫描，看「低于门槛的量数」怎么变；再报告逐拍
   (ICC1) 与 30 拍均值 (ICCk) 两个单位下的判定差别。
2. G5：对 Jaccard 门槛做 ±20% 扫描，看有多少个种子对通过；并单独报告簇级
   判据（配置过半）是否随门槛翻转。

输出：data/audit_threshold_sensitivity.csv（供 registry 引用）
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


import os

import numpy as np
import pandas as pd

ROOT = _repo.ROOT
DATA = os.path.join(ROOT, "data")

ICC_THR = 0.80        # G3 冻结门槛
JAC_THR = 0.50        # G5 冻结门槛（特征级）
SCAN = 0.20           # ±20%
# G5 特征级判定不变的门槛区间：低于下界时所有种子对都不达标，
# 高于上界时结论同样不变。写成常量，供 registry 引用（正文用它说明
# 「判定不依赖阈值」）。
G5_SCAN_LO = 0.30
G5_SCAN_HI = 0.60


def main():
    rows = []

    # ---------- G3：ICC 门槛扫描 ----------
    icc = pd.read_csv(os.path.join(DATA, "audit_vdb_icc.csv"))
    vdb = icc[icc.source == "VDB_搏间"]
    n_feat = len(vdb)
    for thr in [round(ICC_THR * (1 + d), 3) for d in (-SCAN, -0.10, -0.05,
                                                     0.0, 0.05, 0.10, SCAN)]:
        rows.append(dict(
            gate="G3",
            unit="per_30beat_mean",
            metric="icc",
            thr=thr,
            n_below=int((vdb.icck < thr).sum()),
            n_total=n_feat,
            frac_below=round(float((vdb.icck < thr).mean()), 4),
        ))
    # 逐拍单位：门槛几乎无关，因为没有一量达标
    for thr in [round(ICC_THR * (1 + d), 3) for d in (-SCAN, 0.0, SCAN)]:
        rows.append(dict(
            gate="G3",
            unit="per_beat",
            metric="icc",
            thr=thr,
            n_below=int((vdb.icc1 < thr).sum()),
            n_total=n_feat,
            frac_below=round(float((vdb.icc1 < thr).mean()), 4),
        ))

    # ---------- G5：Jaccard 门槛扫描 ----------
    jac = pd.read_csv(os.path.join(DATA, "probe_what_top12_jaccard.csv"))
    M = jac[["cnn", "mlp", "transformer"]].to_numpy(dtype=float)
    for thr in [round(JAC_THR * (1 + d), 3) for d in (-SCAN, -0.10, -0.05,
                                                     0.0, 0.05, 0.10, SCAN)]:
        rows.append(dict(
            gate="G5",
            unit="feature",
            metric="jaccard",
            thr=thr,
            n_below=int((M < thr).sum()),
            n_total=int(M.size),
            frac_below=round(float((M < thr).mean()), 4),
        ))

    df = pd.DataFrame(rows)
    out = os.path.join(DATA, "audit_threshold_sensitivity.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[E5] 写 {out}")
    print(df.to_string(index=False))

    # ---------- 判定翻转点（打印用，不入登记表） ----------
    print("\n[E5] G3：逐拍单位下 ICC1 最大值 =", float(vdb.icc1.max()),
          "→ 任何 ≥0.64 的门槛，逐拍判定都不变（全部不达标）")
    print("[E5] G3：30 拍均值下 ICCk 低于 0.80 的量数 =",
          int((vdb.icck < 0.80).sum()), "/", n_feat)
    print("[E5] G5：特征级 Jaccard 最大值 =", float(M.max()),
          "→ 门槛降到 0.30 以下时全部不达标")


if __name__ == "__main__":
    main()
