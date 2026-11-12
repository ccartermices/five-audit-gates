"""
E-Label：代理标签效度检验
问题：把"高血压"当作"弦脉"，这个标签到底和波形形态有没有关系？
若 AUC≈0.5，则后续任何"中医脉象解码"都是在对噪声建模——这一点必须先证。

设计：
  - 受试者级（每人多段取中位数），n=57
  - 单特征 AUC + bootstrap 95% CI
  - 多变量逻辑回归（L2），分层 5 折，受试者级
  - 对照 1：加入年龄后 AUC 变化（判断是否只是年龄代理）
  - 对照 2：200 次标签置换零分布
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
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

ROOT = _repo.ROOT
R = pd.read_csv(os.path.join(ROOT, "data", "real_features.csv"))
L = pd.read_csv(os.path.join(ROOT, "data", "labels_master.csv"))
L["subject_id"] = L.subject_id.astype(str)
L = L[L.label.notna()]

EXCL = {"n_beats", "rr_cv", "beat_consistency", "chosen_ch", "snr_best", "cyc_dur", "hr"}
FEAT = [c for c in R.columns if c in L.columns or
        (R[c].dtype.kind in "f" and c not in EXCL)]
FEAT = [c for c in FEAT if c in R.columns and R[c].notna().mean() > 0.9]

Sub = R.groupby("sid")[FEAT].median().reset_index().rename(columns={"sid": "subject_id"})
Sub["subject_id"] = Sub.subject_id.astype(str)
D = Sub.merge(L[["subject_id", "label", "age", "sbp"]], on="subject_id")
y = (D.label == "弦脉").astype(int).values
X = D[FEAT].values
print(f"n={len(D)}  弦={y.sum()}  非弦={(1-y).sum()}  features={len(FEAT)}")


def auc_ci(x, y, nboot=2000, seed=0):
    ok = np.isfinite(x)
    x, yy = x[ok], y[ok]
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


print("\n[1] 单特征 AUC（>0.65 或 <0.35 才算有信号）")
rows = []
for j, f in enumerate(FEAT):
    a, lo, hi = auc_ci(X[:, j], y)
    if np.isnan(a):
        continue
    rows.append(dict(feature=f, auc=round(a, 3), lo=round(lo, 3), hi=round(hi, 3),
                     informative=int((lo > 0.5) or (hi < 0.5))))
A = pd.DataFrame(rows).sort_values("auc", ascending=False)
A.to_csv(os.path.join(ROOT, "data", "e_label_single_auc.csv"), index=False)
print(A.head(10).to_string(index=False))
print(f"  有统计信号的形态特征：{int(A.informative.sum())}/{len(A)}")


def cv_auc(Xm, yy, seed=0):
    clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        LogisticRegression(C=0.5, max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(clf, Xm, yy, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(yy, p), p


auc_all, p_all = cv_auc(X, y)
auc_age, _ = cv_auc(D[["age"]].values, y)
auc_sbp, _ = cv_auc(D[["sbp"]].values, y)
auc_morph, _ = cv_auc(np.delete(X, [FEAT.index("cyc_dur")] if "cyc_dur" in FEAT else [], axis=1), y)
print(f"\n[2] 多变量 5 折 CV AUC")
print(f"  仅年龄          : {auc_age:.3f}")
print(f"  仅 SBP          : {auc_sbp:.3f}")
print(f"  全部形态特征    : {auc_all:.3f}")

rng = np.random.RandomState(0)
null = np.array([cv_auc(X, rng.permutation(y))[0] for _ in range(200)])
print(f"\n[3] 标签置换零分布（200 次）：mean={null.mean():.3f} sd={null.std():.3f} "
      f"95%={np.percentile(null,95):.3f}")
pval = (null >= auc_all).mean()
print(f"  实测 AUC={auc_all:.3f}  p={pval:.3f} ->",
      "形态可区分代理标签" if pval < 0.05 else "形态无法区分代理标签（该标签对波形无信息）")

pd.DataFrame([dict(setting="age_only", auc=round(auc_age, 3)),
              dict(setting="sbp_only", auc=round(auc_sbp, 3)),
              dict(setting="morphology_all", auc=round(auc_all, 3)),
              dict(setting="null_mean", auc=round(float(null.mean()), 3)),
              dict(setting="null_p95", auc=round(float(np.percentile(null, 95)), 3)),
              dict(setting="p_value", auc=round(float(pval), 4))]
             ).to_csv(os.path.join(ROOT, "data", "e_label_validity.csv"), index=False)
print("\nsaved e_label_single_auc.csv / e_label_validity.csv")
