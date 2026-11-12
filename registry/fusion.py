"""
跨源融合实验：回答"能不能把真实队列和公库统一起来训练/验证"。

三档协议（缺一档就无法判断融合是否真的有增益）：
  P1 in-domain : 每个源内部受试者级 5 折 —— 该源自身的天花板
  P2 cross     : 单源训练 -> 另一源测试 —— 朴素跨域迁移
  P3 fused     : 多源训练 -> 留出域测试（LODO）—— 融合的真实收益

两个任务（标签语义分层，不混用）：
  T1 SBP 回归 : 仅用标定源（PWDB 仿真真值 / MIMIC 术中测量 / VDB 波形自测）
  T2 HTN 分类 : 仅用"标签独立于波形"的源（VDB 病历诊断 / CM 袖带阈值）
                MIMIC 的 HTN 由同一段 ABP 派生 -> 同源泄漏，禁止进 T2 监督

域对齐三档（用来排除"增益其实是分布对齐带来的"）：
  none / zsrc（训练域统计量）/ zdom（各域自身统计量，transductive）

输出 data/fusion_*.csv
"""

import os as _os, sys as _sys

# Resolved at run time: scripts/ holds sibling groups, the repository root holds data/.
_S = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _S)
for _d in _os.listdir(_S):
    if _d[:1].isdigit() and _os.path.isdir(_os.path.join(_S, _d)):
        _sys.path.append(_os.path.join(_S, _d))
import _repo  # noqa: E402

import os, sys, csv, json, itertools, warnings
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
ROOT = _repo.ROOT
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from harmonize import MORPH, ABS

RNG = 20261031


def rd(p):
    with open(p, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def fl(x, d=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else d
    except (TypeError, ValueError):
        return d


def load():
    rows = [r for r in rd(os.path.join(DATA, "corpus_v2_subject.csv"))
            if int(fl(r["quality"], 1)) == 1]
    X = np.array([[fl(r[f]) for f in MORPH] for r in rows], float)
    d = dict(
        rows=rows, X=X, feats=MORPH,
        dom=np.array([r["dataset"] for r in rows]),
        sid=np.array([r["subject_id"] for r in rows]),
        sbp=np.array([fl(r["sbp"]) for r in rows]),
        dbp=np.array([fl(r["dbp"]) for r in rows]),
        htn=np.array([int(fl(r["y_htn"], 0)) for r in rows]),
        indep=np.array([int(fl(r["y_htn_indep"], 0)) for r in rows]),
        age=np.array([fl(r["age"]) for r in rows]),
    )
    return d


# ---------------------------------------------------------------- 对齐
def align(Xtr, Xte, mode, dtr=None, dte=None):
    if mode == "none":
        return Xtr, Xte
    if mode == "zsrc":
        mu = np.nanmedian(Xtr, axis=0)
        sd = np.nanstd(Xtr, axis=0) + 1e-9
        return (Xtr - mu) / sd, (Xte - mu) / sd
    if mode == "zdom":
        def zn(A, d):
            out = A.copy()
            for u in np.unique(d):
                m = d == u
                mu = np.nanmedian(A[m], axis=0)
                sd = np.nanstd(A[m], axis=0) + 1e-9
                out[m] = (A[m] - mu) / sd
            return out
        return zn(Xtr, dtr), zn(Xte, dte)
    raise ValueError(mode)


def make_model(task, kind):
    if task == "reg":
        est = Ridge(alpha=3.0) if kind == "ridge" else HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_depth=6, l2_regularization=1.0,
            random_state=RNG)
    else:
        est = LogisticRegression(max_iter=2000, C=0.3) if kind == "logit" else \
            HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                           max_depth=6, l2_regularization=1.0,
                                           random_state=RNG)
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()), ("m", est)])


def score(task, y, p):
    if task == "reg":
        return dict(mae=mean_absolute_error(y, p), r2=r2_score(y, p),
                    rho=float(np.corrcoef(y, p)[0, 1]) if len(y) > 2 else np.nan)
    try:
        return dict(auc=roc_auc_score(y, p))
    except ValueError:
        return dict(auc=np.nan)


# ---------------------------------------------------------------- P1 域内
def p1_indomain(d, task, kind, mode="zsrc", nsp=5):
    out = []
    y = d["sbp"] if task == "reg" else d["htn"]
    for ds in np.unique(d["dom"]):
        m = d["dom"] == ds
        if task == "cls" and d["indep"][m].max() == 0:
            continue   # 标签由同段波形派生，禁止进入 T2（同源泄漏）
        if m.sum() < (12 if task == "reg" else 20):
            continue
        if task == "cls" and len(np.unique(y[m])) < 2:
            continue
        Xg, yg = d["X"][m], y[m]
        if task == "reg":
            kf = KFold(nsp, shuffle=True, random_state=RNG)
        else:
            kf = StratifiedKFold(nsp, shuffle=True, random_state=RNG)
        pred = np.full(len(yg), np.nan)
        for tr, te in kf.split(Xg, yg if task == "cls" else None):
            a, b = align(Xg[tr], Xg[te], mode,
                         np.zeros(len(tr)), np.zeros(len(te)))
            mdl = make_model(task, kind).fit(a, yg[tr])
            pred[te] = mdl.predict(b)
        s = score(task, yg, pred)
        s.update(protocol="P1_indomain", train=ds, test=ds, n=int(m.sum()))
        out.append(s)
    return out


# ---------------------------------------------------------------- P2/P3
def p23(d, task, kind, mode="zsrc", fused=True):
    """fused=True -> P3 多源训练留出域；False -> P2 单源训练跨域"""
    y = d["sbp"] if task == "reg" else d["htn"]
    out = []
    doms = [x for x in np.unique(d["dom"]) if np.isfinite(y[d["dom"] == x]).sum() > 0]
    if task == "cls":
        doms = [x for x in doms if d["indep"][d["dom"] == x].max() == 1]
    for tgt in doms:
        te = d["dom"] == tgt
        ok_te = te & np.isfinite(y)
        if ok_te.sum() < (12 if task == "reg" else 20):
            continue
        if task == "cls" and len(np.unique(y[ok_te])) < 2:
            continue
        srcs = [s for s in doms if s != tgt]
        if not fused:
            for s in srcs:
                tr = (d["dom"] == s) & np.isfinite(y)
                if tr.sum() < 12:
                    continue
                a, b = align(d["X"][tr], d["X"][ok_te], mode,
                             d["dom"][tr], d["dom"][ok_te])
                mdl = make_model(task, kind).fit(a, y[tr])
                sc = score(task, y[ok_te], mdl.predict(b))
                sc.update(protocol="P2_cross", train=s, test=tgt, n=int(ok_te.sum()))
                out.append(sc)
        else:
            tr = (d["dom"] != tgt) & np.isfinite(y)
            if task == "cls":
                # 只允许标签独立于波形的源进入 T2 监督
                tr &= (d["indep"] == 1)
            if tr.sum() < 20:
                continue
            a, b = align(d["X"][tr], d["X"][ok_te], mode,
                         d["dom"][tr], d["dom"][ok_te])
            mdl = make_model(task, kind).fit(a, y[tr])
            sc = score(task, y[ok_te], mdl.predict(b))
            sc.update(protocol="P3_fused", train="+".join(sorted(set(d["dom"][tr]))),
                      test=tgt, n=int(ok_te.sum()))
            out.append(sc)
    return out


def wcsv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})


def main():
    d = load()
    print("subjects by domain:",
          {k: int((d["dom"] == k).sum()) for k in np.unique(d["dom"])})
    print("morph features =", len(MORPH))

    # ---- 任务可行性概览
    for ds in np.unique(d["dom"]):
        m = d["dom"] == ds
        s = d["sbp"][m]
        print(f"  {ds:6s} n={m.sum():5d} sbp_med={np.nanmedian(s):6.1f} "
              f"sbp_na={np.isnan(s).sum():4d} htn+={d['htn'][m].sum():4d} "
              f"indep={d['indep'][m].max()}")

    out = []
    # ---------- T1 SBP 回归
    for kind in ("ridge", "gbm"):
        for mode in ("none", "zsrc", "zdom"):
            out += p1_indomain(d, "reg", kind, mode)
            out += p23(d, "reg", kind, mode, fused=False)
            out += p23(d, "reg", kind, mode, fused=True)
    wcsv(os.path.join(DATA, "fusion_T1_sbp.csv"), out)
    print(f"T1 rows = {len(out)}")

    # ---------- T2 HTN 分类
    out2 = []
    for kind in ("logit", "gbm"):
        for mode in ("none", "zsrc"):
            out2 += p1_indomain(d, "cls", kind, mode)
            out2 += p23(d, "cls", kind, mode, fused=False)
            out2 += p23(d, "cls", kind, mode, fused=True)
    wcsv(os.path.join(DATA, "fusion_T2_htn.csv"), out2)
    print(f"T2 rows = {len(out2)}")


if __name__ == "__main__":
    main()
