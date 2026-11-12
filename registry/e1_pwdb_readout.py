"""
E1-B_STIFF 读出（第二版，含校准泄漏对照）

关键设计：PWDB 波形以绝对 mmHg 给出。真实场景里没有动脉置管就没有绝对压力标定，
"从波形估血压"若依赖绝对幅值，就是在读标定值而不是读形态。
因此所有目标都跑两套协议：
    CAL   允许使用绝对压力特征（pp_mmHg / mbp_mmHg）—— 对应"有创/已校准"
    UNCAL 只用无量纲形态特征 —— 对应"指夹/腕带未校准"
若某目标在 UNCAL 下崩塌而 CAL 下近 1.0，则该目标的"高精度"是标定泄漏，不是机制读出。
"""

import os as _os, sys as _sys

# Resolved at run time: scripts/ holds sibling groups, the repository root holds data/.
_S = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _S)
for _d in _os.listdir(_S):
    if _d[:1].isdigit() and _os.path.isdir(_os.path.join(_S, _d)):
        _sys.path.append(_os.path.join(_S, _d))
import _repo  # noqa: E402

import numpy as np, pandas as pd, os, warnings
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

ROOT = _repo.ROOT
F = pd.read_csv(os.path.join(ROOT, "data", "pwdb_features.csv"))
T = pd.read_csv(os.path.join(ROOT, "data", "pwdb_truth.csv"))
df = F.merge(T, on="idx")
df = df[df.plausible == 1].reset_index(drop=True)

TARGETS = {
    "AI_r":   "波形增强指数（反射/僵硬）",
    "PWV_cf": "颈-股脉搏波速（僵硬金标准）",
    "SBP_a":  "中心收缩压",
    "PP_a":   "中心脉压",
    "age":    "年龄",
}
n0 = len(df)
df = df.dropna(subset=list(TARGETS)).reset_index(drop=True)
print(f"n plausible = {n0} -> n targets finite = {len(df)}")

ALL = [c for c in F.columns if c != "idx"]
ABS = [c for c in ALL if c.endswith("_pp_mmHg") or c.endswith("_mbp_mmHg")]
UNCAL = [c for c in ALL if c not in ABS]
HARM = [c for c in UNCAL if (c.startswith("radial_h") and c.endswith("_r"))] + \
       ["radial_hf_energy", "radial_aix_proxy", "radial_cyc_dur"]
print(f"features: all={len(ALL)}  absolute={len(ABS)}  uncal={len(UNCAL)}  harmonic={len(HARM)}")
print("  dropped in UNCAL:", ABS)

T = dict(ridge=make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(1.0)),
         gbm=make_pipeline(SimpleImputer(strategy="median"),
                           HistGradientBoostingRegressor(max_depth=4, max_iter=300,
                                                         learning_rate=0.06, random_state=0)))


def cv_random(cols, y, k=5, seed=0):
    idx = np.random.RandomState(seed).permutation(len(y))
    out = {n: np.full(len(y), np.nan) for n in T}
    for te in np.array_split(idx, k):
        tr = np.setdiff1d(idx, te)
        for n, mdl in T.items():
            mdl.fit(df.iloc[tr][cols], y[tr])
            out[n][te] = mdl.predict(df.iloc[te][cols])
    return out


def cv_extrap(cols, y, pname):
    lvl = df["var_" + pname].values
    tr, te = np.where(lvl <= 0)[0], np.where(lvl == 1)[0]
    out = {}
    for n, mdl in T.items():
        mdl.fit(df.iloc[tr][cols], y[tr])
        out[n] = (mdl.predict(df.iloc[te][cols]), y[te])
    return out


rows = []
for tname in TARGETS:
    y = df[tname].values.astype(float)
    r = {"target": tname, "desc": TARGETS[tname]}
    for pname, cols in [("CAL", ALL), ("UNCAL", UNCAL), ("HARM", HARM)]:
        o = cv_random(cols, y)
        for mdl in ("ridge", "gbm"):
            r[f"{pname}_{mdl}"] = round(float(spearmanr(o[mdl], y)[0]), 3)
    for p in ["pwv", "mbp", "hr"]:
        e = cv_extrap(UNCAL, y, p)
        r[f"UNCAL_extrap_{p}"] = round(float(spearmanr(*e["gbm"])[0]), 3)
    rows.append(r)
    print(f"  {tname:7s} CAL={r['CAL_gbm']:+.3f}  UNCAL={r['UNCAL_gbm']:+.3f}  "
          f"HARM={r['HARM_gbm']:+.3f}  | UNCAL extrap pwv={r['UNCAL_extrap_pwv']:+.2f} "
          f"mbp={r['UNCAL_extrap_mbp']:+.2f} hr={r['UNCAL_extrap_hr']:+.2f}")

R = pd.DataFrame(rows)
R.to_csv(os.path.join(ROOT, "data", "e1_pwdb_readout.csv"), index=False)
print()
print(R[["target", "CAL_gbm", "UNCAL_gbm", "UNCAL_ridge", "HARM_gbm",
         "UNCAL_extrap_pwv", "UNCAL_extrap_mbp", "UNCAL_extrap_hr"]].to_string(index=False))
print()
p = float(R.loc[R.target == "PWV_cf", "UNCAL_gbm"].iloc[0])
print(f"[GATE B_STIFF / UNCAL] PWV_cf rho = {p:+.3f} ->", "PASS (>=0.30)" if p >= 0.30 else "FAIL")
s = float(R.loc[R.target == "SBP_a", "UNCAL_gbm"].iloc[0])
print(f"[对照] SBP_a UNCAL rho = {s:+.3f} -> "
      f"{'形态可读出' if s >= 0.30 else '仅靠绝对幅值，形态不可读出（标定泄漏已证实）'}")
