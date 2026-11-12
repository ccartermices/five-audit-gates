"""
M2: 决定性检验 —— SAE 潜变量到底对齐"生成因子"还是"下游相关量"？

PWDB 的每一例都由已知的物理参数生成（hr/sv/lvet/mbp/pwv/dia/age）。
若潜变量只对齐下游指标（AI_r、SBP）而不对齐生成因子，那么"可解释"只是巧合相关；
若对齐生成因子，则可以做真正的干预实验（改因子 → 改潜变量 → 改形态）。

同时做冗余性定量：按 |rho| 从大到小逐个剔除潜变量，看读出何时崩塌。
"""

import os as _os, sys as _sys

# Resolved at run time: scripts/ holds sibling groups, the repository root holds data/.
_S = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _S)
for _d in _os.listdir(_S):
    if _d[:1].isdigit() and _os.path.isdir(_os.path.join(_S, _d)):
        _sys.path.append(_os.path.join(_S, _d))
import _repo  # noqa: E402

import numpy as np, pandas as pd, os, scipy.io as sio, warnings
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = _repo.ROOT
D = np.load(os.path.join(ROOT, "data", "m1_sae.npz"), allow_pickle=True)
Z = D["Z"]
T = pd.read_csv(os.path.join(ROOT, "data", "pwdb_truth.csv"))

# 与 m1 完全一致的行筛选
ok = (T.plausible.values == 1)
for c in ["AI_r", "PWV_cf", "SBP_a", "PP_a", "age"]:
    ok &= np.isfinite(T[c].values)
Tk = T.iloc[np.where(ok)[0]].reset_index(drop=True)
assert len(Tk) == len(Z), (len(Tk), len(Z))

# ---- 真实生成因子（物理量，不是 -1/0/1 编码）----
m = sio.loadmat(os.path.join(ROOT, "data", "pwdb", "pwdb_data.mat"),
                squeeze_me=True, struct_as_record=False)
c = m["data"].config
FAC = {
    "hr_bpm": c.hr, "sv_mL": c.sv, "lvet_s": c.lvet,
    "mbp_mmHg": c.mbp, "pwv_m_s": c.pwv, "dia_mm": c.dia, "age_yr": c.age,
}
FAC = {k: np.asarray(v, float)[np.where(ok)[0]] for k, v in FAC.items()}
print("生成因子范围：")
for k, v in FAC.items():
    print(f"  {k:10s} {v.min():8.2f} .. {v.max():8.2f}  (uniq {len(np.unique(v))})")

print("\n[1] 潜变量 vs 生成因子（Spearman，只列活跃潜变量）")
rows = []
for k in range(Z.shape[1]):
    if Z[:, k].std() < 1e-6:
        continue
    r = {"latent": k}
    for fn, fv in FAC.items():
        r["f_" + fn] = round(float(spearmanr(Z[:, k], fv)[0]), 3)
    for cname in ["AI_r", "PWV_cf", "SBP_a"]:
        r["d_" + cname] = round(float(spearmanr(Z[:, k], Tk[cname].values)[0]), 3)
    rows.append(r)
A = pd.DataFrame(rows)
A["max_factor"] = A[[c for c in A if c.startswith("f_")]].abs().max(1)
A = A.sort_values("max_factor", ascending=False).reset_index(drop=True)
A.to_csv(os.path.join(ROOT, "data", "m2_factor_alignment.csv"), index=False)
print(A.to_string(index=False))

# ---- 生成因子的可读出性：因子 -> 波形潜变量（有监督探针）----
def cv_probe(Fm, y, k=5, seed=0):
    ii = np.random.RandomState(seed).permutation(len(y))
    out = np.full(len(y), np.nan)
    for te in np.array_split(ii, k):
        tr = np.setdiff1d(ii, te)
        mdl = make_pipeline(StandardScaler(), Ridge(1.0)).fit(Fm[tr], y[tr])
        out[te] = mdl.predict(Fm[te])
    return out

print("\n[2] 生成因子能否从 SAE 潜变量线性读出（5 折 rho）")
rows = []
for fn, fv in FAC.items():
    if len(np.unique(fv)) < 3:
        continue
    rows.append(dict(factor=fn, rho_from_SAE=round(float(spearmanr(cv_probe(Z, fv), fv)[0]), 3)))
    print(f"  {fn:10s} rho={rows[-1]['rho_from_SAE']:+.3f}")
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "data", "m2_factor_readout.csv"), index=False)

# ---- 冗余性：按 |rho(PWV)| 从大到小剔除，看 PWV 读出如何衰减 ----
print("\n[3] 冗余性：逐个剔除潜变量（按 |rho PWV| 降序）")
y = Tk.PWV_cf.values.astype(float)
order = [int(k) for k in A.sort_values("d_PWV_cf", key=lambda s: s.abs(), ascending=False).latent]
curves = []
alive = [k for k in range(Z.shape[1]) if Z[:, k].std() > 1e-6]
base = float(spearmanr(cv_probe(Z, y), y)[0])
print(f"  基线（全 {len(alive)} 个活跃潜变量） rho={base:+.3f}")
for j in range(len(order) + 1):
    drop = order[:j]
    keep = [k for k in alive if k not in drop]
    if not keep:
        curves.append(dict(n_dropped=j, kept=0, rho=np.nan)); break
    r = float(spearmanr(cv_probe(Z[:, keep], y), y)[0])
    curves.append(dict(n_dropped=j, kept=len(keep), rho=round(r, 3),
                       dropped=",".join(map(str, drop))))
    print(f"  剔除 {j:2d} 个（{','.join(map(str,drop)) or '-'}）-> 剩 {len(keep):2d} 个, rho={r:+.3f}")
CV = pd.DataFrame(curves)
CV.to_csv(os.path.join(ROOT, "data", "m2_redundancy.csv"), index=False)

# ---- 单变量必要性：每个潜变量去掉后的 rho 变化 ----
print("\n[4] 单变量必要性（去掉任意一个，rho 变化）")
single = []
for k in alive:
    keep = [j for j in alive if j != k]
    r = float(spearmanr(cv_probe(Z[:, keep], y), y)[0])
    single.append(dict(latent=k, rho_without=round(r, 3), delta=round(r - base, 3)))
SV = pd.DataFrame(single).sort_values("delta")
SV.to_csv(os.path.join(ROOT, "data", "m2_single_necessity.csv"), index=False)
print(SV.to_string(index=False))
print(f"\n  最大单变量损失 = {SV.delta.min():+.3f}  →",
      "存在必要单变量" if SV.delta.min() < -0.05 else "无任何单变量必要（完全分布式编码）")
