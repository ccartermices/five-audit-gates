"""
E-P1 / E-P2（VitalDB）
E-P1 内在维度：PR、PC90/95、PC1 单独的解释力
E-P2 代理标签效度：preop_htn(→弦脉) 能否从形态学区分
     单特征 AUC + bootstrap CI、多变量 GroupKFold CV、置换零分布
"""

import os as _os, sys as _sys

# Resolved at run time: scripts/ holds sibling groups, the repository root holds data/.
_S = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _S)
for _d in _os.listdir(_S):
    if _d[:1].isdigit() and _os.path.isdir(_os.path.join(_S, _d)):
        _sys.path.append(_os.path.join(_S, _d))
import _repo  # noqa: E402

import numpy as np, pandas as pd, os, json, warnings
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

ROOT = _repo.ROOT
F = pd.read_csv(os.path.join(ROOT, "data", "vitaldb_features.csv"))
Z = np.load(os.path.join(ROOT, "data", "vitaldb_cycles.npz"), allow_pickle=True)
X = Z["X"].astype(np.float64)
print(f"records={len(F)} cycles={X.shape}")

# ---------- E-P1 内在维度 ----------
Xn = (X - X.min(1, keepdims=True)) / (X.max(1, keepdims=True) - X.min(1, keepdims=True) + 1e-9)
p = PCA(64, random_state=0).fit(Xn)
ev = p.explained_variance_ratio_
pr = float((ev.sum() ** 2) / (ev ** 2).sum())
pc90 = int(np.searchsorted(np.cumsum(ev), 0.90) + 1)
pc95 = int(np.searchsorted(np.cumsum(ev), 0.95) + 1)
print(f"\n[E-P1] PR={pr:.2f}  PC90={pc90}  PC95={pc95}  ev[:5]={np.round(ev[:5]*100,1)}")

pc1 = p.transform(Xn)[:, 0]
rows = []
for tgt in ["aix_proxy", "h2_r", "sbp_wave", "dbp_wave", "hr", "age"]:
    if tgt not in F.columns:
        continue
    y = F[tgt].values.astype(float)
    ok = np.isfinite(y) & np.isfinite(pc1)
    rows.append(dict(target=tgt, rho_PC1=round(float(spearmanr(pc1[ok], y[ok])[0]), 3)))
E1 = pd.DataFrame(rows)
E1.to_csv(os.path.join(ROOT, "data", "vdb_ep1_dimensionality.csv"), index=False)
print(E1.to_string(index=False))
gate1 = (pr >= 5) and (pc90 >= 8)
print(f"  G0 判据: PR>=5 且 PC90>=8 -> {'PASS' if gate1 else 'FAIL（禁止做多特征分解主张）'}")

# ---------- E-P2 代理标签 ----------
D = F[F.preop_htn.isin([0, 1])].copy().reset_index(drop=True)
EXCL = {"n_beats", "rr_cv", "beat_consistency", "caseid", "cid", "chosen_ch", "snr_best"}
FEAT = [c for c in D.columns
        if D[c].dtype.kind in "f" and c not in EXCL and D[c].notna().mean() > 0.9]
print(f"\n[E-P2] n={len(D)}  弦={int((D.preop_htn==1).sum())}  非弦={int((D.preop_htn==0).sum())}  features={len(FEAT)}")
y = D.preop_htn.values.astype(int)
Xm = D[FEAT].values


def auc_ci(x, yy, nboot=1000, seed=0):
    ok = np.isfinite(x)
    x, yy = x[ok], yy[ok]
    if len(np.unique(yy)) < 2:
        return np.nan, np.nan, np.nan
    a = roc_auc_score(yy, x)
    rng = np.random.RandomState(seed)
    bs = []
    for _ in range(nboot):
        i = rng.randint(0, len(x), len(x))
        if len(np.unique(yy[i])) < 2:
            continue
        bs.append(roc_auc_score(yy[i], x[i]))
    return a, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


rows = []
for j, f in enumerate(FEAT):
    a, lo, hi = auc_ci(Xm[:, j], y)
    if np.isfinite(a):
        rows.append(dict(feature=f, auc=round(a, 3), lo=round(lo, 3), hi=round(hi, 3),
                         informative=int((lo > 0.5) or (hi < 0.5))))
A = pd.DataFrame(rows).sort_values("auc", ascending=False)
A.to_csv(os.path.join(ROOT, "data", "vdb_ep2_single_auc.csv"), index=False)
print(A.head(8).to_string(index=False))
print(f"  有统计信号的形态特征：{int(A.informative.sum())}/{len(A)}")


def cv_auc(Xf, yy, groups, seed=0):
    clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        LogisticRegression(C=0.5, max_iter=2000))
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    pr = cross_val_predict(clf, Xf, yy, cv=cv, groups=groups, method="predict_proba")[:, 1]
    return roc_auc_score(yy, pr)


grp = D.subject_id.astype(str).values

# ---- 特征分组：绝对压 / 形态 / 人口学（协变量）----
ABS = [c for c in FEAT if c.endswith(("_mmHg", "_wave"))]
DEMO = [c for c in FEAT if c in ("age", "bmi", "height", "weight", "asa")]
MORPH = [c for c in FEAT if c not in ABS + DEMO]
print(f"\n  分组: ABS={len(ABS)}  MORPH={len(MORPH)}  DEMO={len(DEMO)}")
print(f"    ABS:  {ABS}")
print(f"    DEMO: {DEMO}")

settings = [
    ("DEMO only", DEMO),
    ("MORPH only (无绝对压)", MORPH),
    ("ABS only (压力本身)", ABS),
    ("MORPH+DEMO", MORPH + DEMO),
    ("ALL", FEAT),
]
res = {}
for name, cols in settings:
    if not cols:
        continue
    a = cv_auc(D[cols].values, y, grp)
    res[name] = a
    print(f"  {name:22s} AUC={a:.3f}  ({len(cols)} 特征)")

# ---- 年龄分层：形态在年龄层内是否还有区分力 ----
D["age_tert"] = pd.qcut(D.age, 3, labels=["年轻", "中年", "老年"])
print("\n  年龄分层内的形态 AUC（判断是否只是年龄代理）")
strat = []
for t, sub in D.groupby("age_tert", observed=True):
    if sub.preop_htn.nunique() < 2 or len(sub) < 60:
        print(f"    {t}: n={len(sub)} 单一类别或样本不足，跳过"); continue
    a = cv_auc(sub[MORPH].values, sub.preop_htn.values.astype(int),
               sub.subject_id.astype(str).values)
    strat.append(dict(stratum=str(t), n=len(sub), auc=round(a, 3)))
    print(f"    {t}: n={len(sub)}  MORPH AUC={a:.3f}")
if strat:
    pd.DataFrame(strat).to_csv(os.path.join(ROOT, "data", "vdb_ep2_age_strata.csv"), index=False)

rng = np.random.RandomState(0)
null = np.array([cv_auc(D[MORPH].values, rng.permutation(y), grp) for _ in range(100)])
pval = float((null >= res["MORPH only (无绝对压)"]).mean())
print(f"\n  置换零分布(100, MORPH): mean={null.mean():.3f} sd={null.std():.3f} "
      f"95%={np.percentile(null,95):.3f}")
print(f"  实测 MORPH AUC={res['MORPH only (无绝对压)']:.3f}  p={pval:.3f}")

gain = res["MORPH+DEMO"] - res["DEMO only"]
print(f"\n  形态相对人口学的增量 AUC = {gain:+.3f}")
print("  G6 判定：",
      "通过" if (pval < 0.05 and res["MORPH only (无绝对压)"] >= 0.65 and gain > 0.02)
      else "不通过（形态不提供独立于协变量的信息）")

pd.DataFrame([dict(setting=k, auc=round(v, 3)) for k, v in res.items()] +
             [dict(setting="null_mean", auc=round(float(null.mean()), 3)),
              dict(setting="null_p95", auc=round(float(np.percentile(null, 95)), 3)),
              dict(setting="p_value_morph", auc=round(pval, 4)),
              dict(setting="delta_morph_over_demo", auc=round(gain, 4)),
              dict(setting="n", auc=len(D)),
              dict(setting="PR", auc=round(pr, 2)),
              dict(setting="PC90", auc=pc90)]
             ).to_csv(os.path.join(ROOT, "data", "vdb_ep2_validity.csv"), index=False)
print("\nsaved vdb_ep1_dimensionality.csv / vdb_ep2_single_auc.csv / vdb_ep2_validity.csv")
