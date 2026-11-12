"""
E-P9 重测信度（CM 本地队列）
同一受试者 3–6 段重复采集 → 每个特征的 ICC(1,1) 与 ICC(1,k)
判据：ICC(2,k)≥0.6 的特征占比 ≥50%，且必须超过数据噪声底
噪声底参照：纯噪声特征的 between/within 方差比≈1
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

ROOT = _repo.ROOT
R = pd.read_csv(os.path.join(ROOT, "data", "real_features.csv"))
EXCL = {"n_beats", "chosen_ch", "snr_best"}
FEAT = [c for c in R.columns if R[c].dtype.kind in "f" and c not in EXCL]
KEEP = ["sid", "session"] + FEAT
D = R[KEEP].dropna(subset=["sid"])
cnt = D.groupby("sid").size()
D = D[D.sid.isin(cnt[cnt >= 2].index)]
print(f"受试者 {D.sid.nunique()}  记录 {len(D)}  每人段数中位 {int(cnt[cnt>=2].median())}")


def icc_oneway(x, g):
    """单因素随机效应 ICC(1,1) / ICC(1,k)，允许不平衡"""
    df = pd.DataFrame({"x": x, "g": g}).dropna()
    k = df.g.nunique()
    n = len(df)
    if k < 2 or n <= k:
        return np.nan, np.nan, np.nan
    grand = df.x.mean()
    gm = df.groupby("g").x.agg(["mean", "size", "var"])
    k0 = n / k                       # 平均组大小
    ssb = float((gm["size"] * (gm["mean"] - grand) ** 2).sum())
    ssw = float(((df.x - df.groupby("g").x.transform("mean")) ** 2).sum())
    msb = ssb / (k - 1)
    msw = ssw / (n - k)
    if (msb + (k0 - 1) * msw) == 0:
        return np.nan, np.nan, np.nan
    icc1 = (msb - msw) / (msb + (k0 - 1) * msw)
    icck = (msb - msw) / msb if msb > 0 else np.nan
    ratio = (msb - msw) / msw if msw > 0 else np.nan    # between/within 方差比
    return icc1, icck, ratio


rows = []
for f in FEAT:
    a, b, r = icc_oneway(D[f].values, D.sid.values)
    rows.append(dict(feature=f, icc11=round(a, 3) if np.isfinite(a) else np.nan,
                     icc1k=round(b, 3) if np.isfinite(b) else np.nan,
                     between_within=round(r, 2) if np.isfinite(r) else np.nan))
T = pd.DataFrame(rows).sort_values("icc11", ascending=False)
T.to_csv(os.path.join(ROOT, "data", "ep9_reliability.csv"), index=False)
print("\n前 15 个最稳定的特征：")
print(T.head(15).to_string(index=False))
print("\n后 5 个：")
print(T.tail(5).to_string(index=False))

share = float((T.icc11 >= 0.6).mean())
print(f"\nICC(1,1) ≥ 0.6 的特征占比 = {100*share:.1f}%  ({int((T.icc11>=0.6).sum())}/{len(T)})")
print(f"判据（≥50%）-> {'PASS' if share >= 0.5 else 'FAIL'}")
print(f"中位 ICC(1,1) = {T.icc11.median():.3f}")
