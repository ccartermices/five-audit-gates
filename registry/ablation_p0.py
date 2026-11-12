"""P0 两项改动的 2×2 消融：savgol（有/无）× p2_rel 定义（q2 / q2÷q1）。

为什么必须消融
--------------
把两项一起改，v3 的 p2_rel 年龄梯度从 +0.324 掉到 −0.087。此时有三种可能：
(1) 去 savgol 让 P2 定位不稳；(2) 改分母放大了 q1 附近的噪声；(3) 两者互相抵消。
不拆开就分不清，而"该保留哪一项"完全取决于这个归因。

判据（事前声明）
----------------
处理链的好坏只有一个硬指标：**同一批人、同一特征，保真的链应该让生物学梯度更清楚**。
动脉硬化随年龄单调变化，AIx / 切迹深度是文献里最稳的两个年龄标志物。
若某组合把这两个梯度弄没了，无论它在单段上看起来多"锐利"，都不能进生产链。

附带诊断：p2_rel 恰为 1.0 的样本，其 i2−i1 若落在平台内（≤2 点），
说明那是量化平顶造成的伪局部极大，而非真实的 P2=P1。
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
import io
import os
import sys
import zipfile

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_preproc as RP          # noqa: E402
from pwfeat import p2_index        # noqa: E402

ROOT = _repo.ROOT
DATA = os.path.join(ROOT, "data")


def load_ages():
    import pandas as pd
    zf = zipfile.ZipFile(RP.ZIP)
    b = io.BytesIO(zf.read("25个channels被试基本信息表/25个channels被试基本信息表.xls"))
    d = pd.read_excel(b, sheet_name="Sheet3", header=None, engine="xlrd")
    out = {}
    for i in range(1, len(d)):
        pid = d.iloc[i, 0]
        if isinstance(pid, str):
            try:
                out[pid.strip()] = float(d.iloc[i, 15])
            except (TypeError, ValueError):
                pass
    return out


def shape_vals(beat):
    """从一个中位搏里取 (p2_old, p2_new, notch, i2-i1, q1)。"""
    i1, i2, jn, q = p2_index(beat)
    q1 = float(q[i1])
    return dict(p2_old=float(q[i2]), p2_new=float(q[i2] / max(q1, 1e-9)),
                notch=float(q[i1] - q[jn]), di=int(i2 - i1), q1=q1)


def main():
    cache = os.path.join(DATA, "ablation_p0.csv")
    if os.environ.get("REPORT_ONLY") == "1" and os.path.exists(cache):
        with open(cache, encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            for k, v in list(r.items()):
                if k in ("subject_id", "segment", "why_s", "why_n"):
                    continue
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    r[k] = np.nan
        print(f"[只读报告] 复用 {cache}（{len(rows)} 行），不重跑前处理")
        report(rows)
        return

    zf = zipfile.ZipFile(RP.ZIP)
    recs = []
    for n in zf.namelist():
        p = n.split("/")
        if (len(p) == 4 and p[3].endswith(".mat")
                and "/Cell Matrix/" in n and not p[3].startswith("Z")):
            recs.append((p[2], p[3][:-4]))
    recs.sort()
    print(f"段数 {len(recs)}，两种链各跑一次", flush=True)

    rows = []
    for i, (sid, seg) in enumerate(recs, 1):
        row = dict(subject_id=sid, segment=seg)
        for tag, sv in (("s", True), ("n", False)):
            try:
                r = RP.preprocess(sid, seg, zf, use_savgol=sv)
            except Exception as e:
                row[f"ok_{tag}"] = 0
                row[f"why_{tag}"] = f"err:{type(e).__name__}"
                continue
            row[f"ok_{tag}"] = int(bool(r["usable"]))
            row[f"why_{tag}"] = r["why"]
            row[f"snr_{tag}"] = round(float(r.get("snr", np.nan)), 3)
            b = r.get("beat")
            if b is None:
                continue
            for k, v in shape_vals(b).items():
                row[f"{k}_{tag}"] = v
        rows.append(row)
        if i % 50 == 0:
            print(f"  {i}/{len(recs)}", flush=True)

    cols, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); cols.append(k)
    with open(os.path.join(DATA, "ablation_p0.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    print(f"-> data/ablation_p0.csv（{len(rows)} 行）")
    report(rows)


def report(rows):
    ages = load_ages()

    def grad(key, okflag, subset=None):
        bysub = {}
        for r in rows:
            if r.get(okflag) != 1:
                continue
            if subset is not None and (r["subject_id"], r["segment"]) not in subset:
                continue
            v = r.get(key, np.nan)
            s = r["subject_id"]
            if isinstance(v, (int, float)) and np.isfinite(v) and s in ages:
                bysub.setdefault(s, []).append(float(v))
        if len(bysub) < 10:
            return np.nan, np.nan, len(bysub)
        x = np.array([np.median(v) for v in bysub.values()])
        a = np.array([ages[s] for s in bysub])
        rho, p = spearmanr(x, a)
        return rho, p, len(x)

    print("\n" + "=" * 78)
    print("表1 四种组合的年龄梯度（按人取中位，n 人）")
    print("=" * 78)
    print(f"{'量/链':<26s}{'savgol':>16s}{'无savgol':>16s}{'n':>5s}")
    for key, col in (("p2_old", "p2_old"), ("p2_new", "p2_new"), ("notch", "notch")):
        line = f"{key:<26s}"
        ns = []
        for tag in ("s", "n"):
            rho, p, n = grad(f"{col}_{tag}", f"ok_{tag}")
            star = "*" if np.isfinite(p) and p < 0.05 else " "
            line += (f"{rho:+.3f}{star}p={p:.2f}").rjust(16)
            ns.append(n)
        print(line + f"{min(ns):>5d}")
    print("（* = p<0.05；p2 的生理方向应为 +（AIx 随年龄升），notch 应为 −（切迹变浅））")

    print("\n" + "=" * 78)
    print("表2 因子效应拆解（同一批人，看梯度是否被破坏）")
    print("=" * 78)
    for key, cold, cnew in (("p2", "p2_old", "p2_new"), ("notch", "notch", "notch")):
        base = grad(f"{cold}_s", "ok_s")[0]
        d_savgol = grad(f"{cold}_n", "ok_n")[0]         # 只动 savgol
        d_def = grad(f"{cnew}_s", "ok_s")[0]            # 只动定义
        both = grad(f"{cnew}_n", "ok_n")[0]             # 两者都动
        print(f"  {key:<5s} 基线 {base:+.3f} | 只去savgol {d_savgol:+.3f} "
              f"| 只改定义 {d_def:+.3f} | 两者都改 {both:+.3f}")

    print("\n" + "=" * 78)
    print("表3 同段一致性（共同可用段）：处理链改动有多大")
    print("=" * 78)
    both_ok = [r for r in rows if r.get("ok_s") == 1 and r.get("ok_n") == 1]
    print(f"两链都可用 {len(both_ok)} 段")
    for a, b, name in [("p2_old_s", "p2_old_n", "只去savgol（p2 旧定义）"),
                       ("p2_old_s", "p2_new_s", "只改p2定义（含savgol）"),
                       ("p2_old_s", "p2_new_n", "两者都改"),
                       ("notch_s", "notch_n", "notch：只去savgol")]:
        x = np.array([r.get(a, np.nan) for r in both_ok], float)
        y = np.array([r.get(b, np.nan) for r in both_ok], float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 20:
            continue
        rho = spearmanr(x[ok], y[ok])[0]
        print(f"  {name:<26s} rho={rho:+.3f}  |Δ|中位={np.median(np.abs(x[ok]-y[ok])):.3f}")

    print("\n" + "=" * 78)
    print("表4 p2_new 恰为 1.0 的样本：是平台伪峰还是真的 P2=P1")
    print("=" * 78)
    for tag, col in (("s", "p2_new_s"), ("n", "p2_new_n")):
        one = [r for r in rows if r.get(f"ok_{tag}") == 1
               and isinstance(r.get(col), (int, float))
               and abs(float(r[col]) - 1.0) < 1e-6]
        di = np.array([r.get(f"di_{tag}", np.nan) for r in one], float)
        di = di[np.isfinite(di)]
        tot = sum(1 for r in rows if r.get(f"ok_{tag}") == 1)
        if len(di):
            print(f"  savgol={'Y' if tag=='s' else 'N'}  ==1.0 共 {len(one)}/{tot}"
                  f" ({len(one)/max(tot,1)*100:.1f}%)   i2−i1: 中位 {np.median(di):.0f}"
                  f"  ≤2 点占 {(di<=2).mean()*100:.0f}%  最大 {di.max():.0f}")
        else:
            print(f"  savgol={'Y' if tag=='s' else 'N'}  无 ==1.0 样本（{tot} 段可用）")
    allq = [r["q1_n"] for r in both_ok if "q1_n" in r]
    if allq:
        allq = np.array(allq)
        print(f"\n  P1 是否就是全局峰：q1==1 占 {(np.abs(allq-1)<1e-9).mean()*100:.1f}%，"
              f"中位 q1={np.median(allq):.3f}  → 若 q1 普遍 <1，说明全局峰常在 P1 之后")


if __name__ == "__main__":
    main()
