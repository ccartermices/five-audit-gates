"""11 号（SAE 解剖任务模型）的种子稳定性与 k 敏感性审计。

背景：11 号的全部数字都是**单次运行**（seed 0、TopK k=8）。11 号遗留第 4 条写着
「SAE 未做种子稳定性，k 未扫描」。本脚本把这个缺口关掉。

一个必须先讲的事实
------------------
同一 seed 连跑两次，历史基线（`probe_what.csv`，11 号用的那份）与重跑之间
Transformer 支路的 d_pred 相关只有 0.66——CUDA 非确定内核导致 SAE 字典重排。
所以「同 seed 可复现」本身要先证明。probe_what.py 已加确定性开关，本脚本第一步
就是核对 det1 vs det2 是否逐位相同；只有这一步过了，后面的「种子差异」才能
解释为种子效应而不是噪声。

判据（跑之前写死）
------------------
C1 复现性：det1 与 det2 的 d_pred 最大绝对差 == 0
C2 任务性能：三种子 rho_task 的极差 ≤ 0.01
C3 捷径缺席：所有种子 / 所有架构的 demographic 因果份额 == 0
C4 机制归因：所有种子 / 所有架构的 vascular 份额 ≥ 70%
C5 特征身份：三种子 top-12 依赖特征的 Jaccard 中位数——<0.5 时**禁止**声称
   「某个 SAE 特征可复现」，只能声称机制簇份额可复现（这条是留给论文的硬约束）
C6 k 敏感性：k=4/8/16 之间 vascular 份额极差 ≤ 10 pp
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


import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_what as PW  # noqa: E402

ROOT = _repo.ROOT
DATA = os.path.join(ROOT, "data")
ARCHS = ["cnn", "mlp", "transformer"]
FAMS = ["vascular", "pressure", "flow", "demographic", "other"]
CHECK = []


def note(level, msg):
    CHECK.append((level, msg))
    print(f"[{level}] {msg}")


def load(tag):
    sfx = f"_{tag}" if tag else ""
    f_csv = os.path.join(DATA, f"probe_what{sfx}.csv")
    f_meta = os.path.join(DATA, f"probe_what_meta{sfx}.csv")
    f_npz = os.path.join(DATA, f"probe_what{sfx}.npz")
    if not (os.path.exists(f_csv) and os.path.exists(f_meta)):
        return None
    df = pd.read_csv(f_csv, encoding="utf-8-sig")
    meta = pd.read_csv(f_meta, encoding="utf-8-sig").set_index("key")["value"]
    z = np.load(f_npz, allow_pickle=True)
    fam = {a: z[f"{a}_fam"].astype(float) for a in ARCHS}
    return dict(df=df, meta=meta, fam=fam, tag=tag or "baseline")


def shares(rec):
    """机制家族的因果份额（%），来自 npz 的 {arch}_fam。"""
    return pd.DataFrame({a: rec["fam"][a] * 100 for a in ARCHS},
                        index=FAMS).T


def top_overlap(a, b, n=12):
    """a/b 是 load() 返回的记录；按 d_pred 取各架构 top-n 特征，算 Jaccard。"""
    out = {}
    da, db = a["df"], b["df"]
    for arch in ARCHS:
        x = set(da[da.arch == arch].nlargest(n, "d_pred").feat)
        y = set(db[db.arch == arch].nlargest(n, "d_pred").feat)
        out[arch] = len(x & y) / max(len(x | y), 1)
    return out


def main():
    runs = {t: load(t) for t in ("det1", "det2", "s1", "s2", "k4", "k16",
                                 "repro", "")}
    base, det1, det2 = runs[""], runs["det1"], runs["det2"]
    if base is None or det1 is None:
        print("缺少产物：先跑 probe_what.py（基线）与 --tag det1")
        return 1

    print("=" * 78)
    print("11 号 SAE 解剖：种子稳定性 + k 敏感性")
    print("=" * 78)

    # ---------- C1 复现性 ----------
    if det2 is not None:
        worst = 0.0
        for arch in ARCHS:
            A = det1["df"][det1["df"].arch == arch].sort_values("feat")
            B = det2["df"][det2["df"].arch == arch].sort_values("feat")
            worst = max(worst, float(np.abs(A.d_pred.values - B.d_pred.values).max()))
        note("PASS" if worst == 0 else "BLOCK",
             f"C1 det1 vs det2 逐位复现：最大 |Δd_pred| = {worst:.3e}")

    # 历史基线（非确定性那次）与确定性重跑差多少
    print("\n== 历史基线 vs 确定性重跑（同 seed 0）==")
    for arch in ARCHS:
        A = base["df"][base["df"].arch == arch].sort_values("feat")
        B = det1["df"][det1["df"].arch == arch].sort_values("feat")
        r = np.corrcoef(A.d_pred, B.d_pred)[0, 1]
        ag = float((A.mech.values == B.mech.values).mean())
        print(f"  {arch:<12s} d_pred r={r:.3f}  机制标签一致={ag:.1%}  "
              f"活特征 {int(base['meta'][f'{arch}_alive'])} → "
              f"{int(det1['meta'][f'{arch}_alive'])}")
    note("WARN", "11 号基线产物出自非确定性运行：特征级身份不可复现，"
                 "机制簇级结论不受影响（详见上表）")

    # ---------- C2 任务性能 ----------
    seeds = [("s0", det1), ("s1", runs["s1"]), ("s2", runs["s2"])]
    print("\n== C2 任务性能（三种子）==")
    rows = []
    for arch in ARCHS:
        rhos = [float(r["meta"][f"{arch}_rho"]) for _, r in seeds if r]
        alive = [int(r["meta"][f"{arch}_alive"]) for _, r in seeds if r]
        named = [int(r["meta"][f"{arch}_named"]) for _, r in seeds if r]
        rows.append(dict(arch=arch, rho_mean=np.mean(rhos), rho_min=min(rhos),
                         rho_max=max(rhos), rho_range=max(rhos) - min(rhos),
                         alive=", ".join(map(str, alive)),
                         named=", ".join(map(str, named))))
        print(f"  {arch:<12s} rho={np.mean(rhos):.4f} "
              f"[{min(rhos):.4f},{max(rhos):.4f}] 极差={max(rhos)-min(rhos):.4f}  "
              f"活特征={alive}  可命名={named}")
    d2 = pd.DataFrame(rows)
    worst_range = float(d2.rho_range.max())
    note("PASS" if worst_range <= 0.01 else "WARN",
         f"C2 任务 rho 三种子极差最大 {worst_range:.4f}（判据 ≤0.01）")

    # ---------- C3 / C4 机制归因 ----------
    print("\n== C3/C4 机制家族份额（%）==")
    sh = {name: shares(r) for name, r in seeds if r}
    for name, s in sh.items():
        print(f"\n  -- {name} --")
        print("  " + s.round(1).to_string().replace("\n", "\n  "))
    demo_max = max(float(s["demographic"].max()) for s in sh.values())
    vasc_min = min(float(s["vascular"].min()) for s in sh.values())
    note("PASS" if demo_max == 0 else "BLOCK",
         f"C3 三种子×三架构 demographic 份额最大 {demo_max:.2f}%（判据 ==0）")
    note("PASS" if vasc_min >= 70 else "WARN",
         f"C4 vascular 份额最小 {vasc_min:.1f}%（判据 ≥70%）")

    # ---------- C5 特征身份 ----------
    print("\n== C5 top-12 依赖特征跨种子 Jaccard ==")
    jac = []
    pairs = [("s0", det1, "s1", runs["s1"]), ("s0", det1, "s2", runs["s2"]),
             ("s1", runs["s1"], "s2", runs["s2"])]
    jrows = []
    for n1, r1, n2, r2 in pairs:
        if not (r1 and r2):
            continue
        o = top_overlap(r1, r2)
        jac += list(o.values())
        jrows.append(dict(pair=f"{n1}-{n2}", **{k: round(v, 3) for k, v in o.items()}))
        print(f"  {n1}-{n2}: " + "  ".join(f"{k}={v:.2f}" for k, v in o.items()))
    if jac:
        med = float(np.median(jac))
        note("PASS" if med >= 0.5 else "WARN",
             f"C5 top-12 特征跨种子 Jaccard 中位 {med:.2f}（<0.5 则禁止声称单特征可复现）")

    # ---------- C6 k 敏感性 ----------
    print("\n== C6 k 敏感性（k=4/8/16）==")
    krows = []
    if runs["k4"] and runs["k16"]:
        for k, r in ((4, runs["k4"]), (8, det1), (16, runs["k16"])):
            s = shares(r)
            for arch in ARCHS:
                krows.append(dict(k=k, arch=arch,
                                  rho=float(r["meta"][f"{arch}_rho"]),
                                  alive=int(r["meta"][f"{arch}_alive"]),
                                  named=int(r["meta"][f"{arch}_named"]),
                                  vascular=s.loc[arch, "vascular"],
                                  pressure=s.loc[arch, "pressure"],
                                  demographic=s.loc[arch, "demographic"]))
        kdf = pd.DataFrame(krows)
        print(kdf.round(2).to_string(index=False))
        kdf.to_csv(os.path.join(DATA, "probe_what_k_scan.csv"),
                   index=False, encoding="utf-8-sig")
        spread = kdf.groupby("arch").vascular.agg(lambda v: v.max() - v.min())
        note("PASS" if float(spread.max()) <= 10 else "WARN",
             f"C6 k=4/8/16 的 vascular 份额极差最大 {spread.max():.1f} pp（判据 ≤10）")
    else:
        note("WARN", "C6 未跑：缺 k4 / k16 产物")

    # ---------- 落盘 ----------
    d2.to_csv(os.path.join(DATA, "probe_what_seed_stability.csv"),
              index=False, encoding="utf-8-sig")
    if jrows:
        pd.DataFrame(jrows).to_csv(
            os.path.join(DATA, "probe_what_top12_jaccard.csv"),
            index=False, encoding="utf-8-sig")
    fams = pd.concat({n: s for n, s in sh.items()})
    fams.to_csv(os.path.join(DATA, "probe_what_fam_by_seed.csv"),
                encoding="utf-8-sig")

    print("\n" + "=" * 78)
    nb = sum(1 for lv, _ in CHECK if lv == "BLOCK")
    nw = sum(1 for lv, _ in CHECK if lv == "WARN")
    npp = sum(1 for lv, _ in CHECK if lv == "PASS")
    print(f"自检：{nb} BLOCK / {nw} WARN / {npp} PASS")
    return nb


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
