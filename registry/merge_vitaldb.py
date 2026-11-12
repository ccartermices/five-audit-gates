"""合并 VitalDB 各分片，接上病例标签（preop_htn 等），产出规范要求的四个文件"""

import os as _os, sys as _sys

# Resolved at run time: scripts/ holds sibling groups, the repository root holds data/.
_S = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _S)
for _d in _os.listdir(_S):
    if _d[:1].isdigit() and _os.path.isdir(_os.path.join(_S, _d)):
        _sys.path.append(_os.path.join(_S, _d))
import _repo  # noqa: E402

import numpy as np, pandas as pd, glob, os, json, hashlib

ROOT = _repo.ROOT
VD = os.path.join(ROOT, "data", "vitaldb")

feats, cycs, ids = [], [], []
for f in sorted(glob.glob(os.path.join(VD, "part*_feat.csv"))):
    d = pd.read_csv(f)
    feats.append(d)
    c = f.replace("_feat.csv", "_cyc.npz")
    if os.path.exists(c):
        z = np.load(c)
        cycs.append(z["X"])
        ids.append(z["caseid"])

F = pd.concat(feats, ignore_index=True)
ok = F[F.ok == 1].reset_index(drop=True)
X = np.vstack(cycs) if cycs else np.zeros((0, 256))
cid = np.concatenate(ids) if ids else np.zeros(0)
print(f"records {len(F)} | ok {len(ok)} ({100*len(ok)/max(1,len(F)):.1f}%) | cycles {X.shape}")

# 对齐：cyc 的顺序就是 ok 的顺序，做一次校验
assert len(cid) == len(ok), (len(cid), len(ok))
assert (cid == ok.caseid.values).all(), "caseid 顺序不一致"

C = pd.read_csv(os.path.join(VD, "cases.csv"))
keep = ["caseid", "subjectid", "age", "sex", "height", "weight", "bmi", "asa",
        "preop_htn", "preop_dm", "department", "optype", "ane_type", "death_inhosp", "icu_days"]
M = ok.merge(C[[c for c in keep if c in C.columns]], on="caseid", how="left")

# 规范要求的统一列
M["dataset"] = "VDB"
M["subject_id"] = "VDB:" + M.subjectid.astype("Int64").astype(str)
M["record_id"] = M.caseid.astype(str)
M["site"] = "radial_arterial"
M["unit"] = "mmHg"
M["calibrated"] = 1
M["fs_native"] = 500
M.to_csv(os.path.join(ROOT, "data", "vitaldb_features.csv"), index=False, encoding="utf-8-sig")

# 标签长表（dx_code）
lab = pd.DataFrame(dict(
    dataset="VDB", subject_id=M.subject_id, record_id=M.record_id,
    label=M.preop_htn.map({1: "弦脉", 0: "非弦脉"}),
    label_type="tcm_pulse", is_proxy=1, label_source="preop_htn(dx)->taut",
    confidence="中（病历诊断映射到弦脉仍为代理）",
    sbp=np.nan, dbp=np.nan, age=M.age, sex=M.sex,
    n_sessions=1, note="VitalDB preop_htn 映射；血压取波形自身测量，见 features 表 sbp_wave/dbp_wave"))
lab.to_csv(os.path.join(ROOT, "data", "vitaldb_labels.csv"), index=False, encoding="utf-8-sig")

np.savez_compressed(os.path.join(ROOT, "data", "vitaldb_cycles.npz"),
                    X=X.astype(np.float32), caseid=cid,
                    subject_id=M.subject_id.values,
                    preop_htn=M.preop_htn.values, age=M.age.values)

h = hashlib.sha256(open(os.path.join(ROOT, "scripts", "pwfeat.py"), "rb").read()).hexdigest()
mani = dict(dataset="VDB", n_records=int(len(F)), n_ok=int(len(ok)),
            ok_rate=round(float(len(ok) / max(1, len(F))), 4),
            fs_native=500, nph=256, feature_impl_sha256=h,
            reason_counts=F[F.ok == 0].reason.astype(str).str.split("=").str[0].value_counts().to_dict(),
            quality_gates=dict(beat_consistency_min=0.80, rr_cv_max=0.25, min_dur_s=60))
json.dump(mani, open(os.path.join(ROOT, "data", "vitaldb_manifest.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(json.dumps(mani, ensure_ascii=False, indent=2))
print("\n标签分布：")
print(M.preop_htn.value_counts(dropna=False).to_string())
