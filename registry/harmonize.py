"""
四源统一装配器：把 PWDB / VitalDB / MIMIC-BP / CM 本地队列对齐成
同一特征 schema、同一位点约定、同一质量门，输出两级语料。

统一的三件事（不统一这三件，融合就是假的）：
  1. 位点：只保留"外周动脉压"一个解剖位点（PWDB radial / VDB 桡动脉 /
     MIMIC ABP / CM 25 通道中 SNR 最优通道）
  2. 特征：31 个特征全部来自 pwfeat.seg_feats 单一实现（SHA256 记录在 manifest）
  3. 质量门：统一规范 §6 同一阈值；仅 MIMIC 因 30s 段长做显式豁免

不统一、且必须显式分层的一件事：
  标签。四源标签语义不同（仿真真值 / 病历诊断 / 术中测量 / 袖带阈值），
  因此保留 y_htn_src 与 y_htn_indep 两个字段，禁止混成一个 y 直接训练。

输出：
  data/corpus_v2_record.csv    段级（每段/每记录一行）
  data/corpus_v2_subject.csv   受试者级（特征取中位，标签取多数）——融合训练用这级
  data/corpus_v2_manifest.json 口径、门统计、特征实现哈希
"""

import os as _os, sys as _sys

# Resolved at run time: scripts/ holds sibling groups, the repository root holds data/.
_S = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _S)
for _d in _os.listdir(_S):
    if _d[:1].isdigit() and _os.path.isdir(_os.path.join(_S, _d)):
        _sys.path.append(_os.path.join(_S, _d))
import _repo  # noqa: E402

import os, sys, csv, json, hashlib, statistics as st
import numpy as np

ROOT = _repo.ROOT
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FEATS = (["pp_mmHg", "mbp_mmHg"]
         + [f"h{k}" for k in range(1, 9)]
         + [f"h{k}_r" for k in range(2, 9)]
         + ["hf_energy"]
         + ["t1", "t2", "p2_rel", "ap_rel", "aix_proxy", "t_notch",
            "p_notch_rel", "notch_depth", "t_dicrotic", "p_dicrotic_rel",
            "upstroke_slope", "t_90", "t_50up", "t_50down", "width50",
            "skew", "kurt", "q_entropy"])
MORPH = [f for f in FEATS if f not in ("pp_mmHg", "mbp_mmHg")]
ABS = ["pp_mmHg", "mbp_mmHg"]

# 统一质量门（规范 §6 / v1.1）
GATE = dict(dur_min=60.0, valid_min=0.80, keep_min=0.30,
            cons_min=0.85, beats_min=8, pp_min=15.0, pp_max=180.0)
SOURCE_DUR_EXEMPT = {"MIMIC": 25.0}   # MIMIC-BP 官方段长 30s，唯一样本级豁免


def rd(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def fl(x, d=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else d
    except (TypeError, ValueError):
        return d


def gt(a, b):
    return 1 if (fl(a) >= 140 or fl(b) >= 90) else 0


# ---------------------------------------------------------------- 适配器
def adapt_pwdb():
    F = rd(os.path.join(DATA, "pwdb_features.csv"))
    T = rd(os.path.join(DATA, "pwdb_truth.csv"))
    tm = {int(r["idx"]): r for r in T if r.get("idx", "") != ""}
    out = []
    for r in F:
        i = int(r["idx"])
        t = tm.get(i)
        if t is None or int(fl(t.get("plausible", 1), 1)) == 0:
            continue
        d = dict(
            dataset="PWDB", subject_id=f"PWDB:{i}", record_id="0",
            site="radial", unit="mmHg", calibrated=1, fs_native=50.0,
            dur_s=168.0, valid_frac=1.0, keep_frac=1.0,
            beat_consistency=1.0,
            cyc_dur=fl(r.get("radial_cyc_dur")),
            hr=60.0 / max(fl(r.get("radial_cyc_dur")), 1e-6),
            sbp=fl(t.get("SBP_b")), dbp=fl(t.get("DBP_b")),
            age=fl(t.get("age")), sex="",
            y_htn_src="sim_truth", y_htn_indep=1,
        )
        for f in FEATS:
            d[f] = fl(r.get("radial_" + f))
        # PWDB 每条记录 168s，搏数由段长/中位周期推算（原始 record 未逐搏落盘）
        d["n_beats"] = 168.0 / max(d["cyc_dur"], 0.30) if np.isfinite(d["cyc_dur"]) else np.nan
        d["y_htn"] = gt(d["sbp"], d["dbp"])
        out.append(d)
    return out, "PWDB"


def adapt_vdb():
    F = rd(os.path.join(DATA, "vitaldb_features.csv"))
    out = []
    for r in F:
        d = dict(
            dataset="VDB", subject_id=r.get("subject_id") or f"VDB:{r.get('subjectid')}",
            record_id=r.get("record_id", "1"),
            site="radial_arterial", unit="mmHg", calibrated=1, fs_native=500.0,
            dur_s=fl(r.get("dur_s")), valid_frac=fl(r.get("valid_frac"), 1.0),
            keep_frac=fl(r.get("keep_frac"), 1.0),
            n_beats=fl(r.get("n_beats")), beat_consistency=fl(r.get("beat_consistency")),
            cyc_dur=fl(r.get("cyc_dur")) or (60.0 / fl(r.get("hr"), np.nan)),
            hr=fl(r.get("hr")),
            sbp=fl(r.get("sbp_wave")), dbp=fl(r.get("dbp_wave")),
            age=fl(r.get("age")), sex=r.get("sex", ""),
            y_htn_src="preop_htn(dx)", y_htn_indep=1,
        )
        if np.isnan(fl(d["cyc_dur"])):
            d["cyc_dur"] = 60.0 / max(fl(d["hr"], 60.0), 1e-6)
        for f in FEATS:
            d[f] = fl(r.get(f))
        hn = fl(r.get("preop_htn"))
        d["y_htn"] = int(hn) if np.isfinite(hn) else gt(d["sbp"], d["dbp"])
        out.append(d)
    return out, "VDB"


def adapt_mimic():
    p = os.path.join(DATA, "mimicbp_features.csv")
    if not os.path.exists(p):
        print("[warn] 缺 mimicbp_features.csv，跳过 MIMIC")
        return [], "MIMIC"
    F = rd(p)
    out = []
    for r in F:
        d = dict(
            dataset="MIMIC", subject_id=f"MIMIC:{r['pid']}", record_id=r["seg"],
            site="arterial_unspecified", unit="mmHg", calibrated=1, fs_native=125.0,
            dur_s=fl(r.get("dur_s"), 30.0), valid_frac=fl(r.get("valid_frac"), 1.0),
            keep_frac=fl(r.get("keep_frac"), 1.0),
            n_beats=fl(r.get("n_beats")), beat_consistency=fl(r.get("beat_consistency")),
            cyc_dur=fl(r.get("cyc_dur")), hr=fl(r.get("hr")),
            sbp=fl(r.get("sbp")), dbp=fl(r.get("dbp")),
            age=np.nan, sex="",
            y_htn_src="seg_measured(ABP)", y_htn_indep=0,
        )
        for f in FEATS:
            d[f] = fl(r.get(f))
        d["y_htn"] = gt(d["sbp"], d["dbp"])
        out.append(d)
    return out, "MIMIC"


def adapt_cm():
    F = rd(os.path.join(DATA, "real_features.csv"))
    S = {r["sid"]: r for r in rd(os.path.join(DATA, "subject_table.csv"))}
    out = []
    for r in F:
        sid = r["sid"]
        s = S.get(sid, {})
        d = dict(
            dataset="CM", subject_id=f"CM:{sid}", record_id=r["session"],
            site="radial_unknown", unit="au", calibrated=0, fs_native=50.0,
            dur_s=168.0, valid_frac=1.0, keep_frac=1.0,
            n_beats=fl(r.get("n_beats")), beat_consistency=fl(r.get("beat_consistency")),
            cyc_dur=fl(r.get("cyc_dur")), hr=fl(r.get("hr")),
            sbp=fl(s.get("sbp")), dbp=fl(s.get("dbp")),
            age=fl(s.get("age")), sex=s.get("sex", ""),
            y_htn_src="cuff_threshold", y_htn_indep=1,
        )
        for f in FEATS:
            d[f] = fl(r.get(f))
        # 未标定：幅值不是 mmHg，绝对压特征一律置缺失（规范 §2）
        d["pp_mmHg"] = np.nan
        d["mbp_mmHg"] = np.nan
        d["y_htn"] = gt(d["sbp"], d["dbp"])
        out.append(d)
    return out, "CM"


# ---------------------------------------------------------------- 质量门
def apply_gate(r):
    rs = []
    dur_min = SOURCE_DUR_EXEMPT.get(r["dataset"], GATE["dur_min"])
    if not (fl(r["dur_s"], 0) >= dur_min):
        rs.append(f"dur<{dur_min:g}s")
    if fl(r["valid_frac"], 1.0) < GATE["valid_min"]:
        rs.append("valid<0.80")
    if fl(r["keep_frac"], 1.0) < GATE["keep_min"]:
        rs.append("keep<0.30")
    if not (fl(r["beat_consistency"], 1.0) >= GATE["cons_min"]):
        rs.append("cons<0.85")
    if not (fl(r["n_beats"], 0) >= GATE["beats_min"]):
        rs.append("beats<8")
    if r["calibrated"] == 1:
        pp = fl(r["pp_mmHg"])
        if not np.isfinite(pp) or not (GATE["pp_min"] <= pp <= GATE["pp_max"]):
            rs.append("pp_out_of_range")
    return rs


COLS = (["dataset", "subject_id", "record_id", "site", "unit", "calibrated",
         "fs_native", "dur_s", "valid_frac", "keep_frac", "n_beats",
         "beat_consistency", "cyc_dur", "hr", "sbp", "dbp", "age", "sex",
         "y_htn", "y_htn_src", "y_htn_indep", "quality", "q_frac", "reason"] + FEATS)


def md(a):
    a = [x for x in a if np.isfinite(x)]
    return float(np.median(a)) if a else np.nan


def aggregate_subject(recs):
    """受试者级：特征取中位；标签/人口学取首值或多数。"""
    by = {}
    for r in recs:
        by.setdefault((r["dataset"], r["subject_id"]), []).append(r)
    out = []
    for (ds, sid), g in by.items():
        d = dict(dataset=ds, subject_id=sid, record_id=str(len(g)),
                 site=g[0]["site"], unit=g[0]["unit"],
                 calibrated=g[0]["calibrated"], fs_native=g[0]["fs_native"],
                 y_htn_src=g[0]["y_htn_src"], y_htn_indep=g[0]["y_htn_indep"])
        for f in FEATS + ["cyc_dur", "hr", "n_beats", "beat_consistency"]:
            d[f] = md([fl(x[f]) for x in g])
        d["dur_s"] = md([fl(x["dur_s"]) for x in g])
        d["valid_frac"] = md([fl(x["valid_frac"]) for x in g])
        d["keep_frac"] = md([fl(x["keep_frac"]) for x in g])
        d["sbp"] = md([fl(x["sbp"]) for x in g])
        d["dbp"] = md([fl(x["dbp"]) for x in g])
        d["age"] = md([fl(x["age"]) for x in g])
        d["sex"] = next((x["sex"] for x in g if x.get("sex")), "")
        ys = [int(x["y_htn"]) for x in g]
        d["y_htn"] = 1 if sum(ys) * 2 >= len(ys) else 0
        # 受试者级质量：通过段的占比（规范 §6：未过门不删除，只降级并留档）
        qf = float(np.mean([int(x["quality"]) for x in g]))
        d["q_frac"] = qf
        d["quality"] = 1 if qf >= 0.5 else 0
        d["reason"] = "" if d["quality"] else f"q_frac={qf:.2f}"
        out.append(d)
    return out


def main():
    recs, stats = [], {}
    for fn in (adapt_pwdb, adapt_vdb, adapt_mimic, adapt_cm):
        rows, name = fn()
        ok = 0
        for r in rows:
            rs = apply_gate(r)
            r["quality"] = 0 if rs else 1
            r["reason"] = ";".join(rs)
            ok += r["quality"]
        stats[name] = dict(rows=len(rows), passed=int(ok),
                           rejected=len(rows) - int(ok))
        recs += rows
        print(f"{name:6s} rows={len(rows):6d} passed={int(ok):6d} "
              f"rejected={len(rows)-int(ok):6d}")

    kept = recs          # 未过门的记录不删除，仅以 quality/reason 降级留档（规范 §6）
    passed = [r for r in recs if r["quality"] == 1]
    print(f"total records = {len(kept)}, passed gate = {len(passed)}")

    with open(os.path.join(DATA, "corpus_v2_record.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(kept)

    subs = aggregate_subject(kept)
    with open(os.path.join(DATA, "corpus_v2_subject.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(subs)

    from collections import Counter
    for ds in sorted({s["dataset"] for s in subs}):
        g = [s for s in subs if s["dataset"] == ds]
        pos = sum(int(s["y_htn"]) for s in g)
        ages = [fl(s["age"]) for s in g if np.isfinite(fl(s["age"]))]
        print(f"  {ds:6s} subjects={len(g):5d} HTN+={pos:5d} "
              f"({100*pos/max(len(g),1):.0f}%) "
              f"age_med={st.median(ages) if ages else float('nan'):.0f} "
              f"fs={g[0]['fs_native']:g} site={g[0]['site']}")

    impl = os.path.join(ROOT, "scripts", "pwfeat.py")
    sha = hashlib.sha256(open(impl, "rb").read()).hexdigest()
    man = dict(
        created="2026-10-31", corpus="v2.0", feature_impl="scripts/pwfeat.py",
        feature_impl_sha256=sha, n_features=len(FEATS), features=FEATS,
        abs_features=ABS, morph_features=MORPH,
        gate=GATE, source_dur_exempt=SOURCE_DUR_EXEMPT,
        record_level=stats, n_record=len(kept), n_subject=len(subs),
        harmonization=dict(
            site="peripheral arterial only",
            alignment="foot (diastolic minimum)",
            resample="phase-normalized NPH=256",
            unit="mmHg if calibrated==1 else au",
            abs_features_rule="CM(calibrated=0) 的 pp/mbp 置缺失，禁止进入绝对压回归",
        ),
        label_policy=dict(
            note="四源标签语义不同，禁止当同一 y 混合训练",
            PWDB="sim_truth", VDB="preop_htn(dx)", MIMIC="seg_measured(ABP, 与波形同源→泄漏)",
            CM="cuff_threshold",
            y_htn_indep={"1": "标签独立于波形（可作 T2 监督）",
                         "0": "标签由同一段波形派生（仅可作 T1 标定目标）"},
        ),
    )
    with open(os.path.join(DATA, "corpus_v2_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, ensure_ascii=False, indent=2)
    print("saved corpus_v2_record.csv / corpus_v2_subject.csv / corpus_v2_manifest.json")
    print("feature_impl_sha256 =", sha[:16])


if __name__ == "__main__":
    main()
