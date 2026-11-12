# -*- coding: utf-8 -*-
"""E2 外部验证：把「同一队列内交叉验证」换成「跨库测试」。

为什么必须做
------------
医学刊对「同一队列内 5 折交叉验证」的容忍度很低。审稿人会问：这个模型在
别的医院、别的设备、别的人群上还成立吗？本稿原来的全部性能数字都来自
同一队列内部的折外预测，所以这个问题在原稿里没有答案。

数据来源
--------
`data/fusion_T1_sbp.csv` 是 `scripts/fusion.py` 的产物，结构固定：
每 20 行一个块，块内依次是 4 行 P1_indomain（同库内训练测试）、12 行
P2_cross（单源训练 → 异源测试）、4 行 P3_fused（多源训练 → 留出域测试）。
块顺序是 (kind, mode) 的嵌套：ridge/none、ridge/zsrc、ridge/zdom、
gbm/none、gbm/zsrc、gbm/zdom。

主口径
------
两个函数的默认参数都是 ridge + zsrc（训练域统计量归一化），所以**主口径取
ridge/zsrc 块**；其余五块作为稳健性区间一并报出，因为不同归一化方式下
跨库结论是否稳定，比任何单块的数字都更重要。

这里有一处必须写进论文的限定
----------------------------
fusion 流水线用的是受试者级形态量与岭回归，与正文的任务模型（三架构深度
网络）不是同一个估计器。所以本实验回答的是「跨库迁移这件事本身有多难」，
不是「本文这个网络跨库表现如何」。论文里必须这样写，否则是把一个估计器的
性质挂到另一个估计器头上。

输出：data/audit_external_validation.csv（供 registry 引用）
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

BLOCK = 20                       # 每个 (kind, mode) 块 20 行
N_BLOCK = 6
BLOCK_NAMES = ["ridge/none", "ridge/zsrc", "ridge/zdom",
               "gbm/none", "gbm/zsrc", "gbm/zdom"]
PRIMARY = 1                      # ridge/zsrc：两个函数的默认口径


def main():
    d = pd.read_csv(os.path.join(DATA, "fusion_T1_sbp.csv"))
    assert len(d) == BLOCK * N_BLOCK, f"行数 {len(d)} 与预期 {BLOCK * N_BLOCK} 不符"

    rows = []
    for b in range(N_BLOCK):
        blk = d.iloc[b * BLOCK:(b + 1) * BLOCK]
        for proto in ("P1_indomain", "P2_cross", "P3_fused"):
            sub = blk[blk.protocol == proto]
            if sub.empty:
                continue
            rows.append(dict(
                block=BLOCK_NAMES[b],
                is_primary=int(b == PRIMARY),
                protocol=proto,
                n_rows=int(len(sub)),
                rho_mean=round(float(sub.rho.mean()), 4),
                rho_min=round(float(sub.rho.min()), 4),
                rho_max=round(float(sub.rho.max()), 4),
                mae_mean=round(float(sub.mae.mean()), 4),
                mae_min=round(float(sub.mae.min()), 4),
                mae_max=round(float(sub.mae.max()), 4),
            ))

    df = pd.DataFrame(rows)
    out = os.path.join(DATA, "audit_external_validation.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[E2] 写 {out}")
    print(df.to_string(index=False))

    # 主口径的逐方向明细（供正文与补充材料引用）
    prim = d.iloc[PRIMARY * BLOCK:(PRIMARY + 1) * BLOCK]
    p1 = prim[prim.protocol == "P1_indomain"]
    p2 = prim[prim.protocol == "P2_cross"]
    print(f"\n[E2] 主口径 {BLOCK_NAMES[PRIMARY]}：")
    print(f"[E2] 同库内（P1）ρ 范围 {p1.rho.min():.3f}–{p1.rho.max():.3f}，"
          f"MAE 范围 {p1.mae.min():.2f}–{p1.mae.max():.2f} mmHg")
    print(f"[E2] 跨库（P2，{len(p2)} 个方向）ρ 范围 {p2.rho.min():.3f}–{p2.rho.max():.3f}，"
          f"中位数 {p2.rho.median():.3f}；MAE 范围 {p2.mae.min():.2f}–{p2.mae.max():.2f} mmHg")
    print(f"[E2] 跨库中 ρ > 0.5 的方向数 = {int((p2.rho > 0.5).sum())}/{len(p2)}")


if __name__ == "__main__":
    main()
