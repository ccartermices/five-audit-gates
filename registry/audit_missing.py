"""L1 缺失结构审计：填补是否伪化了"四源同号"这个主干结论。

要回答
------
1. 四个源里，哪些物理量有缺失、缺多少。
2. `build_l1` 的列中位数填补把边际 rho 挪动了多少。
3. 真正的 pairwise 统计下，四源同号是几个（12 号称 1/33）。
4. `reflection` 簇（弦）的 PC1 是否已经退化成"P2 是否检出"的指示变量 ——
   若是，则 CM/MIMIC 上该簇的份额不是波反射强度，而是设备/算法能否定位反射波峰。

为什么必须查第 4 点
------------------
p2_rel / ap_rel / aix_proxy 在 P2 未检出时置 NaN（16 号）。填补后，这些行的取值
恒等于列中位数，于是整簇的方差结构变成"是否被填补"的两峰。P2 检出率又与可训练性
同序（PWDB 100% / VDB 99.8% / CM 76.5% / MIMIC 74.8%），若不排除这条通路，
"CM 学不动"与"CM 反射簇份额异常"就会被同一个装置伪迹同时解释，而不是两个独立证据。
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
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phys_layer as PL           # noqa: E402
import probe_mediator as PM       # noqa: E402

ROOT = _repo.ROOT
DATA = os.path.join(ROOT, "data")
SRC = ["PWDB", "VDB", "MIMIC", "CM"]
EFF = 0.10          # 判定"有方向"的效应量下界（p 值在 n>2000 时无意义）


def impute(V):
    return np.where(np.isfinite(V), V, np.nanmedian(V, 0))


def rho_pw(V, y, qn):
    """逐量边际 Spearman，真 pairwise（缺失行独立剔除）。"""
    out = {}
    for j, q in enumerate(qn):
        x = V[:, j]
        ok = np.isfinite(x) & np.isfinite(y)
        out[q] = (float(spearmanr(x[ok], y[ok])[0]) if ok.sum() > 30 else np.nan,
                  int(ok.sum()))
    return out


def main():
    os.environ.setdefault("CM_VER", "v3")
    D, Y, G, PF, QN, Vimp = {}, {}, {}, {}, None, {}
    for s in SRC:
        d = PM.load_source(s, keep_nan=True)
        V, y, qn = d["V"].astype(np.float64), d["y"], d["qnames"]
        QN = qn
        D[s], Y[s], G[s] = V, y, d["groups"]
        PF[s] = d["aux"]["p2_found"]
        Vimp[s] = impute(V)
        print(f"  {s:<6s} n={len(y):<6d} 受试者={d['n_group']:<4d} "
              f"P2 检出 {np.nanmean(PF[s]) * 100:5.1f}%", flush=True)

    # ---------------------------------------------------------------- 1 缺失率
    print("\n" + "=" * 78)
    print("§1 逐量缺失率（%）")
    print(f"{'量':<16s}{'簇':<12s}" + "".join(f"{s:>9s}" for s in SRC))
    for j, q in enumerate(QN):
        rate = [100 * (1 - np.isfinite(D[s][:, j]).mean()) for s in SRC]
        if max(rate) > 0.01:
            print(f"{q:<16s}{PL.cluster_of(q):<12s}"
                  + "".join(f"{r:>8.1f}%" for r in rate))
    tot = {s: 100 * (1 - np.isfinite(D[s]).mean()) for s in SRC}
    print("整表缺失：" + "  ".join(f"{s} {tot[s]:.2f}%" for s in SRC))

    # 退化值：P2 未检出时这几个量并不缺，而是塌成常数
    print("\n退化常数（P2 未检出时塌成定值的列）")
    for s in SRC:
        f = PF[s] == 0
        if f.sum() == 0:
            print(f"  {s:<6s} 无未检出段")
            continue
        dg = []
        for q in ("t2", "t_notch", "p_notch_rel"):
            if q not in QN:
                continue
            j = QN.index(q)
            dg.append(f"{q}={'t1' if q.startswith('t') else '1.0'}"
                      f"{(np.abs(D[s][f, j] - (D[s][f, QN.index('t1')] if q.startswith('t') else 1.0)) < 1e-9).mean() * 100:.0f}%")
        print(f"  {s:<6s} 未检出 {f.sum():>4d}/{len(f)} 段：" + "  ".join(dg))

    # ---------------------------------------------------------------- 2 填补偏移
    print("\n" + "=" * 78)
    print("§2 列中位数填补对边际 rho 的影响（只列被填补的量）")
    R, RI = {}, {}
    for s in SRC:
        R[s] = rho_pw(D[s], Y[s], QN)
        RI[s] = rho_pw(Vimp[s], Y[s], QN)
    aff = [q for q in QN if any(not np.isfinite(D[s][:, QN.index(q)]).all() for s in SRC)]
    print(f"{'量':<16s}" + "".join(f"{s + '真/填补':>16s}" for s in SRC))
    for q in aff:
        line = f"{q:<16s}"
        for s in SRC:
            a, b = R[s][q][0], RI[s][q][0]
            line += f"{a:>+7.3f}/{b:>+7.3f}".rjust(16)
        print(line)

    # ---------------------------------------------------------------- 3 同号
    print("\n" + "=" * 78)
    print("§3 四源同号：真 pairwise vs 填补")
    for tag, RR in (("填补（12 号口径）", RI), ("真 pairwise", R)):
        full = [q for q in QN if all(np.isfinite(RR[s][q][0]) for s in SRC)]
        same = [q for q in full if len({RR[s][q][0] > 0 for s in SRC}) == 1]
        det = [q for q in full if all(abs(RR[s][q][0]) >= EFF for s in SRC)]
        same_det = [q for q in det if len({RR[s][q][0] > 0 for s in SRC}) == 1]
        print(f"  {tag:<16s} 四源齐 {len(full):>2d}  同号 {len(same):>2d}"
              f"   |rho|>={EFF:.2f} 的 {len(det):>2d} 个中同号 {len(same_det):>2d}")
        if same:
            print("     同号：" + "、".join(same))

    # ---------------------------------------------------------------- 4 方向一致性
    print("\n" + "=" * 78)
    print("§4 跨源方向一致性：rho 向量之间的 Spearman（真 pairwise）")
    print(f"{'':<10s}" + "".join(f"{s:>9s}" for s in SRC))
    for a in SRC:
        line = f"{a:<10s}"
        for b in SRC:
            ks = [q for q in QN if np.isfinite(R[a][q][0]) and np.isfinite(R[b][q][0])]
            r = spearmanr([R[a][q][0] for q in ks], [R[b][q][0] for q in ks])[0]
            line += f"{r:>+9.3f}"
        print(line)
    print("\n按簇拆（真 pairwise，只算两源都齐全的子向量）")
    print(f"{'簇':<12s}" + "".join(f"{s:>9s}" for s in SRC))
    for c in PL.TABLE:
        qs = [q for q in QN if PL.cluster_of(q) == c]
        if len(qs) < 2:
            continue
        line = f"{c:<12s}"
        for b in SRC:
            a = "VDB"
            if b == a:
                line += f"{'-':>9s}"
                continue
            ks = [q for q in qs if np.isfinite(R[a][q][0]) and np.isfinite(R[b][q][0])]
            if len(ks) < 3:
                line += f"{'n/a':>9s}"
                continue
            line += f"{spearmanr([R[a][q][0] for q in ks], [R[b][q][0] for q in ks])[0]:>+9.3f}"
        print(line)

    # ---------------------------------------------------------------- 5 伪迹检验
    print("\n" + "=" * 78)
    print("§5 reflection 簇的 PC1 是不是「P2 是否检出」的指示变量")
    print("   （|rho| 接近 1 说明该簇维度测的是能不能定位反射波峰，不是反射强度）")
    for s in SRC:
        if PF[s].std() < 1e-9:
            print(f"  {s:<6s} P2 恒检出，无对照")
            continue
        for tag, VV in (("填补", Vimp[s]), ("仅检出段", D[s][PF[s] == 1])):
            if tag == "仅检出段":
                P, cs = PM.cluster_dims(VV, QN)
                fl = PF[s][PF[s] == 1]
            else:
                P, cs = PM.cluster_dims(VV, QN)
                fl = PF[s]
            k = cs.index("reflection")
            r = spearmanr(P[:, k], fl)[0]
            # PC1 主载荷
            idx = [j for j, q in enumerate(QN) if PL.cluster_of(q) == "reflection"]
            A = VV[:, idx]
            A = (A - np.nanmean(A, 0)) / (np.nanstd(A, 0) + 1e-12)
            A = np.nan_to_num(A)
            _, _, vt = np.linalg.svd(A, full_matrices=False)
            top = idx[int(np.argmax(np.abs(vt[0])))]
            print(f"  {s:<6s} {tag:<8s} n={len(fl):<6d} "
                  f"corr(PC1, P2检出)={r:>+6.3f}  主载荷={QN[top]}")

    # ---------------------------------------------------------------- 落盘
    with open(os.path.join(DATA, "l1_rho_pairwise.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "cluster"] + [f"{s}_rho" for s in SRC]
                   + [f"{s}_rho_imputed" for s in SRC] + [f"{s}_n" for s in SRC])
        for q in QN:
            w.writerow([q, PL.cluster_of(q)]
                       + [round(R[s][q][0], 4) for s in SRC]
                       + [round(RI[s][q][0], 4) for s in SRC]
                       + [R[s][q][1] for s in SRC])
    print("\n-> data/l1_rho_pairwise.csv")


if __name__ == "__main__":
    main()
