"""
PWDB 特征提取（改用共享实现 pwfeat，保证与真实队列/VitalDB 同源）
输出：data/pwdb_cycles.npz（相位归一化周期）、data/pwdb_truth.csv、data/pwdb_features.csv
"""

import os as _os, sys as _sys

# Resolved at run time: scripts/ holds sibling groups, the repository root holds data/.
_S = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _S)
for _d in _os.listdir(_S):
    if _d[:1].isdigit() and _os.path.isdir(_os.path.join(_S, _d)):
        _sys.path.append(_os.path.join(_S, _d))
import _repo  # noqa: E402

import numpy as np, pandas as pd, scipy.io as sio, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pwfeat import seg_feats, resample_cycle

ROOT = _repo.ROOT
MAT = os.path.join(ROOT, "data", "pwdb", "pwdb_data.mat")
NPH = 256

print("[1/3] loading mat ...")
m = sio.loadmat(MAT, squeeze_me=True, struct_as_record=False)
d = m["data"]; w, c = d.waves, d.config
fs = int(w.fs)

sites = ["Radial", "Brachial", "Carotid"]
X, CYC = {}, {}
for s in sites + ["PPG_Radial"]:
    raw = getattr(w, "P_" + s) if not s.startswith("PPG") else w.PPG_Radial
    X[s] = np.vstack([resample_cycle(a, NPH) for a in raw])
    CYC[s] = np.array([len(np.asarray(a, float).ravel()) / fs for a in raw])
print("   wave matrix", X["Radial"].shape)

h = d.haemods
H = pd.DataFrame({f: [getattr(x, f) for x in h] for f in h[0]._fieldnames})
H["idx"] = np.arange(len(H)); H["age_cfg"] = c.age
H["plausible"] = d.plausibility.plausibility_log.astype(int)
for i, nm in enumerate(c.variations.param_names):
    H["var_" + str(nm)] = c.variations.params[:, i]
H.to_csv(os.path.join(ROOT, "data", "pwdb_truth.csv"), index=False)
print("   truth", H.shape, "| plausible", int(H.plausible.sum()))

print("[2/3] features (shared impl) ...")
rows = []
for s in sites:
    fr = [seg_feats(x) for x in X[s]]
    rows.append(pd.DataFrame(fr).add_prefix(s.lower() + "_"))
    print("   ", s, "done")
F = pd.concat(rows, axis=1)
for s in sites + ["PPG_Radial"]:
    F[s.lower() + "_cyc_dur"] = CYC[s]
F.insert(0, "idx", np.arange(len(F)))
F.to_csv(os.path.join(ROOT, "data", "pwdb_features.csv"), index=False)
print("   features", F.shape)

np.savez_compressed(os.path.join(ROOT, "data", "pwdb_cycles.npz"),
                    X_rad=X["Radial"].astype(np.float32),
                    X_bra=X["Brachial"].astype(np.float32),
                    X_car=X["Carotid"].astype(np.float32),
                    X_ppg=X["PPG_Radial"].astype(np.float32),
                    fs=fs, age=H.age.values, plausible=H.plausible.values)

print("[3/3] 自检：AIx 代理 vs haemods 真值")
from scipy.stats import spearmanr
sub = H.plausible == 1
for s in sites:
    a = F[f"{s.lower()}_aix_proxy"].values[sub]
    b = H["AI_" + s[0].lower()].values[sub]
    ok = np.isfinite(a) & np.isfinite(b)
    print(f"   {s:8s} rho={spearmanr(a[ok], b[ok])[0]:+.3f} (n={ok.sum()})")
