"""
统一标签表构建
规则（用户指定）：本地真实队列中，凡高血压者标注为「弦脉」，否则「非弦脉」。
这是**代理标签（proxy label）**，不是医师判读，必须在表里显式标记来源与可信度。
同时把连续量（SBP/DBP/年龄）与协变量一并落表，供后续去混杂。

输出 data/labels_master.csv
列：dataset, subject_id, record_id, label, label_type, is_proxy, label_source,
    confidence, sbp, dbp, age, sex, note
"""

import os as _os, sys as _sys

# Resolved at run time: scripts/ holds sibling groups, the repository root holds data/.
_S = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _S)
for _d in _os.listdir(_S):
    if _d[:1].isdigit() and _os.path.isdir(_os.path.join(_S, _d)):
        _sys.path.append(_os.path.join(_S, _d))
import _repo  # noqa: E402

import numpy as np, pandas as pd, os

ROOT = _repo.ROOT
S = pd.read_csv(os.path.join(ROOT, "data", "subject_table.csv"))
S["sid"] = S.sid.astype(str).str.strip()

rows = []
for _, r in S.iterrows():
    sbp, dbp = r.sbp, r.dbp
    if not np.isfinite(sbp):
        lab, note = np.nan, "无血压，无法代理标注"
    elif (sbp >= 140) or (dbp >= 90):
        lab, note = "弦脉", "ProxyRule: SBP>=140 or DBP>=90"
    else:
        lab, note = "非弦脉", "ProxyRule: SBP<140 and DBP<90"
    rows.append(dict(dataset="CM_local", subject_id=r.sid, record_id="",
                     label=lab, label_type="tcm_pulse", is_proxy=1,
                     label_source="HTN_threshold",
                     confidence="低（无医师判读，仅为血压阈值代理）",
                     sbp=sbp, dbp=dbp, age=r.age, sex=r.sex,
                     n_sessions=r.n_sessions, note=note))

L = pd.DataFrame(rows)
L.to_csv(os.path.join(ROOT, "data", "labels_master.csv"), index=False, encoding="utf-8-sig")
print("标签分布：")
print(L.label.value_counts(dropna=False).to_string())
print(f"\n受试者 {len(L)} | 有标签 {(L.label.notna()).sum()} | 弦脉 {(L.label=='弦脉').sum()}")
print("\n年龄/SBP 分层：")
print(L.groupby("label")[["age", "sbp", "dbp"]].agg(["mean", "std", "count"]).round(1).to_string())
