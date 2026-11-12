"""医生盲判标注分析管线（协议 计划/医生标注协议_v1.0_20261103.md §5 的实现）。

用法
----
    python analyze_physician.py --forms form_A.csv form_B.csv
表格式 = physician_annotation_form.csv（pulse_no + 医生判读(弦/非弦) + 置信度）。

预注册判据（协议 §5，判前冻结；与脚本实现一一对应）：
  S1 医生间一致性（2 张表时）：Cohen's κ。
     κ<0.4  -> 共识不足：只报一致性，跳过 S3/S4（诚实失败优于勉强结论）；
     0.4-0.6-> 共识中等：仅用双医生**一致**子集做 S3/S4；
     >=0.6  -> 共识良好：全集做 S3/S4，分歧例标记。
  S2 共识标签：双医生一致 -> 该标签；不一致 -> 分歧例（不进判别分析）。
  S3 AI vs 医生：预标注一致率 + κ，按 AI 置信（高/低）分层报告；
     注意：AI 预标注与 L1 同源，此结果不构成独立证据（仅分歧例挖掘）。
  S4 外部效度锚点（主分析）：L1 30 量（人级中位数）-> 医生标签
     a) 单量 AUC（全部 30 量，Mann-Whitney U 即 AUC）；
     b) 多元：留一人 LOOCV logistic（ridge 正则 C=0.1，标准化在训练折内），
        报 AUC + 1000 次自助 95% CI；**预期不稳**（56 人 x 30 量，维度灾难，
        19 号已证 ridge29 ρ 摆动 -0.11~+0.19），AUC 区间必须与单量并排报告，
        不得单报多元点估计。
  S5 措辞：任何结果按 21 号口径 "consistent with, but not establishing"。

SMOKE 模式：--smoke 用 AI 预标注生成假医生表跑通全管线，产物文件名带
_smoke，仅验证代码路径，数值不作数、不得写入任何报告。
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


import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phys_layer as PL
import probe_mediator as PM

FEATS = sorted({f for c in PL.TABLE.values() for f in c["feats"]})
THR_K_LOW, THR_K_OK = 0.4, 0.6      # S1 预注册门


def cohens_kappa(a, b):
    a = np.asarray(a, object); b = np.asarray(b, object)
    ok = (a != "") & (b != "")
    a, b = a[ok], b[ok]
    if len(a) == 0:
        return np.nan, 0
    po = np.mean(a == b)
    labs = sorted(set(a) | set(b))
    pe = sum(np.mean(a == l) * np.mean(b == l) for l in labs)
    k = (po - pe) / (1 - pe) if pe < 1 else np.nan
    return float(k), int(len(a))


def auc_mw(pos, neg):
    """Mann-Whitney U / (n1*n2) = AUC。pos/neg: 该量的两组取值。"""
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if len(pos) < 3 or len(neg) < 3:
        return np.nan
    u = sum((1 * (pos[i] > neg)).sum() + 0.5 * (pos[i] == neg).sum()
            for i in range(len(pos)))
    return float(u / (len(pos) * len(neg)))


def loocv_auc(X, y, C=0.1, seed=0):
    """留一人 LOOCV logistic，返回 OOF 决策值的 AUC 与逐折预测。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    n = len(y)
    oof = np.full(n, np.nan)
    for i in range(n):
        tr = np.arange(n) != i
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
        clf.fit(sc.transform(X[tr]), y[tr])
        oof[i] = clf.decision_function(sc.transform(X[i:i + 1]))[0]
    ok = np.isfinite(oof)
    return auc_mw(oof[ok][y[ok] == 1], oof[ok][y[ok] == 0]), oof


def boot_ci(X, y, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        v, _ = loocv_auc(X[idx], y[idx])
        if np.isfinite(v):
            vals.append(v)
    return np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forms", nargs="+", default=[])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    tag = "_smoke" if a.smoke else ""

    # ---- 读医生表 ----
    if a.smoke:
        pre = pd.read_csv(os.path.join(PM.DATA, "ai_prelabel_cm.csv"),
                          encoding="utf-8-sig", index_col=0)
        sid_col = pre.index.astype(str)
        fill = pre.prelabel.map({"弦": "弦", "非弦": "非弦"})
        fill = fill.where(fill.notna(),
                          pd.Series(np.where(pre.R1.values == "弦", "弦", "非弦"),
                                    index=pre.index))
        forms = [pd.DataFrame({"pulse_no": sid_col, "判读": fill.values,
                               "置信度": np.where(pre.confidence == "高", 3, 1)})]
        forms[0].columns = ["pulse_no", "判读", "置信度"]
        print("[SMOKE] 用 AI 预标注充当假医生表，数值不作数")
    else:
        forms = []
        for f in a.forms:
            d = pd.read_csv(f, encoding="utf-8-sig")
            d.columns = ["pulse_no", "判读", "置信度", "备注"][: len(d.columns)]
            if d["判读"].isna().any():
                raise SystemExit(f"{f} 有未填判读（协议 §4 要求全部 56 例）")
            forms.append(d)
    if not forms:
        raise SystemExit("未提供医生表：--forms 或 --smoke")

    sid_key = "pulse_no"
    merged = forms[0][[sid_key, "判读"]].rename(columns={"判读": "doc1"})
    if len(forms) > 1:
        merged = merged.merge(forms[1][[sid_key, "判读"]].rename(
            columns={"判读": "doc2"}), on=sid_key)
        k, n = cohens_kappa(merged.doc1, merged.doc2)
        print(f"S1 医生间 κ = {k:+.3f} (n={n})  门: {THR_K_LOW}/{THR_K_OK}")
        if k < THR_K_LOW:
            print("判定：共识不足 -> 按协议只报一致性，跳过 S3/S4")
            _save(merged, k, None, None, None, tag)
            return
        agree = merged.doc1 == merged.doc2
        if k < THR_K_OK:
            print("判定：共识中等 -> 仅用双医生一致子集")
            merged = merged[agree]
        merged["label"] = np.where(merged.doc1 == "弦", 1, 0)
    else:
        merged["label"] = np.where(merged.doc1 == "弦", 1, 0)
    print(f"标签分布: 弦={merged.label.sum()} 非弦={(1-merged.label).sum()} "
          f"(n={len(merged)})")
    if merged.label.nunique() < 2:
        print("判定：单类标签，判别分析无意义 -> 只报一致性")
        _save(merged, None, None, None, None, tag)
        return

    # ---- S3 AI vs 医生 ----
    pre = pd.read_csv(os.path.join(PM.DATA, "ai_prelabel_cm.csv"),
                      encoding="utf-8-sig", index_col=0)
    pre2 = pre.reset_index().rename(columns={pre.index.name or "index": "pulse_no"})
    pre2["pulse_no"] = pre2["pulse_no"].astype(str)
    merged[sid_key] = merged[sid_key].astype(str)
    # 只取预标注信息列，不带特征列（否则与 S4 的 agg 列冲突被重命名 _x/_y）
    keep = [c for c in ("pulse_no", "prelabel", "confidence", "R1", "R2",
                        "s1_reflect", "s2_slide") if c in pre2.columns]
    m2 = merged.merge(pre2[keep], on="pulse_no", how="left")
    # AI 三分类 vs 医生二分类：端位例（弦/非弦）算一致率；「边界」是 AI 拒判，
    # 不进一致率，单独报告拒判例中医生的分布（拒判例正是困难例）。
    ai_end = m2.prelabel.map({"弦": "弦", "非弦": "非弦"})
    ok = ai_end.notna()
    k_ai, n_ai = cohens_kappa(ai_end[ok], m2.doc1[ok])
    agree_rate = float((ai_end[ok] == m2.doc1[ok]).mean()) if ok.any() else np.nan
    print(f"S3 AI vs 医生: 端位一致率={agree_rate:.3f} κ={k_ai:+.3f} "
          f"(n={n_ai}, AI拒判 {int((~ok).sum())} 例另报)")
    rej = m2[~ok]
    if len(rej):
        print(f"   AI拒判例中医生分布: {rej.doc1.value_counts().to_dict()} "
              f"（接近均匀=拒判合理；一边倒=规则漏掉可判信息）")
    for lv in ("高", "低"):
        mml = m2[m2.confidence == lv]
        if len(mml):
            ae = ai_end[mml.index]
            if ae.notna().any():
                print(f"   AI置信={lv}: n={len(mml)} "
                      f"一致率={float((ae == mml.doc1).mean()):.3f}")
            else:
                print(f"   AI置信={lv}: n={len(mml)} 一致率=不适用（AI 全部拒判）")
    disc = m2[ok & (ai_end != m2.doc1)][["pulse_no", "doc1", "prelabel"]]
    print(f"   分歧例 {len(disc)} 个（L1 与医生判断的分歧点，讨论章节材料）")

    # ---- S4 L1 -> 医生标签 ----
    from pwfeat import seg_feats
    d = PM.load_source("CM")
    sids = pd.Series(d["meta"]["sid"])
    F = pd.DataFrame([seg_feats(x) for x in d["X"].astype(float)])
    F["sid"] = sids.values
    agg = F.groupby("sid").median(numeric_only=True)
    agg.index = agg.index.astype(str)
    mm = merged.merge(agg, left_on="pulse_no", right_index=True, how="inner")
    print(f"S4 对齐后 n={len(mm)}")
    if len(mm) < 10 or mm.label.nunique() < 2:
        print("样本不足，跳过判别")
        _save(merged, None, (agree_rate, k_ai), disc, None, tag)
        return
    Xa = mm[FEATS].apply(pd.to_numeric, errors="coerce")
    Xa = Xa.fillna(Xa.median()).values
    ya = mm.label.values.astype(int)
    rows = []
    for f in FEATS:
        v = auc_mw(mm.loc[ya == 1, f].values.astype(float),
                   mm.loc[ya == 0, f].values.astype(float))
        rows.append(dict(kind="单量", feat=f, value=round(v, 4) if np.isfinite(v) else ""))
    v, oof = loocv_auc(Xa, ya)
    lo, hi = boot_ci(Xa, ya, B=300)
    rows.append(dict(kind="多元LOOCV", feat=f"ridge30(C=0.1,n={len(mm)})",
                     value=round(v, 4) if np.isfinite(v) else ""))
    print(f"S4 多元 LOOCV AUC = {v:+.3f} [{lo:+.3f},{hi:+.3f}]（SMOKE 不作数）"
          if a.smoke else f"S4 多元 LOOCV AUC = {v:+.3f} [{lo:+.3f},{hi:+.3f}]")
    top = sorted([r for r in rows if r["kind"] == "单量" and r["value"] != ""],
                 key=lambda r: abs(float(r["value"]) - 0.5), reverse=True)[:8]
    for r in top:
        print(f"   单量 {r['feat']:<12s} AUC={r['value']}")
    _save(merged, None, (agree_rate, k_ai), disc,
          dict(rows=rows, lo=lo, hi=hi, k_doc=None), tag)


def _save(merged, k_doc, ai_res, disc, s4, tag):
    out = os.path.join(PM.DATA, f"physician_anchor_results{tag}.csv")
    rows = []
    if k_doc is not None:
        rows.append(dict(step="S1_医生间κ", key="kappa", value=round(k_doc, 4)))
    if ai_res is not None:
        rows.append(dict(step="S3_AI_vs_医生", key="一致率", value=round(ai_res[0], 4)))
        rows.append(dict(step="S3_AI_vs_医生", key="kappa", value=round(ai_res[1], 4)))
    if disc is not None and len(disc):
        for _, r in disc.iterrows():
            rows.append(dict(step="S3_分歧例", key=r["pulse_no"],
                             value=f"医生={r['doc1']} AI={r['prelabel']}"))
    if s4 is not None:
        for r in s4["rows"]:
            rows.append(dict(step="S4_判别", key=f"{r['kind']}:{r['feat']}",
                             value=r["value"]))
        rows.append(dict(step="S4_自助区间", key="多元LOOCV",
                         value=f"[{s4['lo']:.3f},{s4['hi']:.3f}]"))
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
