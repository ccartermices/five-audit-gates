# -*- coding: utf-8 -*-
"""可解释性五性质审计 —— 把「模型依据什么」从描述升级为干预。

现有可解释性证据只有一条腿：SAE 簇份额（描述性、相关性）。本脚本补五种性质，
每一种都给一个可检验的量，而不是形容词：

  F 充分性  30 个 L1 量能把黑箱输出重建到什么程度；几个量就够（简约曲线）
  N 必要性  在潜空间整簇消掉 SAE 特征，输出掉多少
  S 特异度  生理编辑 vs 传感器噪声编辑，模型响应是否分离（模型侧为主证据）
  D 方向性  剂量-反应单调，方向可读
  C 收敛性  三种互不依赖的归因方法在**簇**层面是否给同一排序
  U 不确定性 簇份额的自助区间

外加跨域检查：同样的算子在 CM 真实脉搏周期上，是否让 L1 层朝**同一方向**移动
（L1 层不是仿真器专属）。

三条不能违反的设计约束
----------------------
1. 干预必须在**波形**上做，不在派生量上做。直接改 Vm 会造出非物理的「特征向量」，
   模型读到的东西没有任何设备能产生，结论不可外推。
2. 干预必须落在**同一台模型**上。见 probe_what.run_arch 返回 model/sae/st 的注释。
3. **算子必须先验证自身不含高频泄漏**。首版用 σ≈0.02 周期的高斯核，结果在真实
   CM 周期上四个算子全部以 device 簇（h5–h8）为首 —— 那是核的旁瓣，不是生理。
   修法：Hann 隆起（两端值为 0、一阶导也为 0）再截断到 6 次谐波。此过程保留在
   26 号文档里，因为它正是「干预实验如何凭空造出一个设备伪迹结论」的实例。

关于 descriptor 族
------------------
任何形态编辑都必然移动 spectrum/shape 族（h1–h4、skew、kurt），因为 descriptor
按定义就是形态的统计重述。这不是缺陷，是分类法使然。因此特异度判据不要求
「只有一个簇动」，而要求：生理算子的响应剖面彼此相似、且与噪声算子的剖面分离；
真正的判决放在模型侧 —— 模型对生理编辑与对纯高频噪声的响应是否量级不同。
"""
from __future__ import annotations

import os as _os, sys as _sys

# Resolved at run time: scripts/ holds sibling groups, the repository root holds data/.
_S = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _S)
for _d in _os.listdir(_S):
    if _d[:1].isdigit() and _os.path.isdir(_os.path.join(_S, _d)):
        _sys.path.append(_os.path.join(_S, _d))
import _repo  # noqa: E402


import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = _repo.SCRIPTS
ROOT = _repo.ROOT
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, HERE)

import pwfeat  # noqa: E402
import probe_what as P  # noqa: E402
import torch  # noqa: E402

# ------------------------------------------------------------------ L1 定义
L1 = json.load(open(os.path.join(DATA, "phys_mediator_v1.json"), encoding="utf-8"))
CLUSTERS = list(L1["table"].keys())
F2C = {f: c for c, t in L1["table"].items() for f in t["feats"]}
F2FAM = {f: t["family"] for c, t in L1["table"].items() for f in t["feats"]}
L1_FEATS = list(F2C)
FAMS = ["mechanism", "descriptor", "device"]

# ------------------------------------------------------------------ 形态算子
# 这些常量故意放在模块级：Methods 里每个数都要能被登记表按名字取走（op: const），
# 藏在字典字面量里就改了也没人知道。
OP_LATE_OFFSET = 0.06      # 重搏波算子的锚点偏移（周期比例）
OP_UPSTROKE_FRAC = 0.45    # 上升支算子的锚点位置（收缩峰前的比例）
OP_HANN_MAX = 0.16         # 最宽核（切迹）
OP_HANN_MID = 0.14         # 中等核（重搏波、反射波）
OP_HANN_MIN = 0.12         # 最窄核（上升支）
OP_KMAX = 6                # 谐波截断
OP_DOSE_MAX = 0.15         # 机制算子最大剂量（周期峰峰值比例）
NOISE_DOSE_MAX = 0.12      # 噪声算子最大剂量
DOSE_BOOT = 400            # 剂量-反应区间的自助重采样次数

KMAX = OP_KMAX

OPS = {
    "notch": dict(target="reflection", w=OP_HANN_MAX, sign=-1.0, where="jn",
                  gloss="切迹加深/填平"),
    "late": dict(target="dicrotic", w=OP_HANN_MID, sign=+1.0, where="jn+off",
                 gloss="舒张早期重搏波抬升"),
    "reflect": dict(target="reflection", w=OP_HANN_MID, sign=+1.0, where="i2",
                    gloss="收缩期反射波 P2 增强"),
    "upstroke": dict(target="stiffness", w=OP_HANN_MIN, sign=+1.0, where="peak",
                     gloss="上升支前段抬高"),
    "noise": dict(target="noise", w=0.0, sign=+1.0, where="band",
                  gloss="h5–h8 高频注入（负对照）"),
}

# 七档对称网格，由 OP_DOSE_MAX 生成；与旧版硬写的 [-0.15…+0.15] 逐位相同。
DOSE_GRID = [round(OP_DOSE_MAX * f, 4)
             for f in (-1.0, -2 / 3, -1 / 3, 0.0, 1 / 3, 2 / 3, 1.0)]
DOSES = {
    "notch": DOSE_GRID, "late": DOSE_GRID, "reflect": DOSE_GRID,
    "upstroke": DOSE_GRID,
    "noise": [0.0, 0.02, 0.04, 0.08, NOISE_DOSE_MAX],
}


def hann_kern(n, c, w, kmax=KMAX):
    """两端值为 0、一阶导为 0 的隆起，再截断到 kmax 次谐波。

    截断保证算子不向 h5–h8 注入能量（device 族）。不截断时，窄高斯核的旁瓣会把
    四个生理算子全判成「设备伪迹」——这是本脚本踩过的坑。
    """
    t = np.arange(n) / max(n - 1, 1)
    g = np.zeros(n)
    m = np.abs(t - c) <= w
    g[m] = 0.5 * (1.0 + np.cos(np.pi * (t[m] - c) / w))
    if g.max() <= 0:
        return g
    F = np.fft.rfft(g)
    F[kmax + 1:] = 0.0
    g = np.fft.irfft(F, n)
    s = np.abs(g).max()
    return g / s if s > 1e-12 else g


def apply_op(q, op, a, phase=None):
    """在归一化单周期 q∈[0,1] 上施加形态算子。返回 (q_edit, clipped_frac)。

    边界余量渐变（taper）：编辑量在 q 接近 0 或 1 处衰减到 0，因此不越界。
    首版直接截断到 [0,1]，噪声算子在 4–13% 的采样点上被削平 —— 那是往波形里
    另塞了一个平顶形变，会把"对高频噪声的响应"和"对削顶的响应"混在一起。
    taper 只影响幅值范围的上下 15%，不改变编辑的方向与位置。
    """
    q = np.asarray(q, float)
    n = len(q)
    if a == 0.0:
        return q.copy(), 0.0
    i1, i2, jn, _ = pwfeat.p2_index(q)
    taper = np.clip(np.minimum(q, 1.0 - q) / 0.15, 0.0, 1.0)
    if op == "noise":
        t = np.arange(n) / n
        ph = np.zeros(4) if phase is None else phase
        # 除以 4 而不是 2：四个正弦相加的峰值是 4，归一到 |s|<=1，
        # 否则剂量 0.12 时峰值可达 0.24，边界渐变也救不回来（实测 3.8% 越界）。
        s = sum(np.sin(2 * np.pi * (5 + j) * t + ph[j]) for j in range(4)) / 4.0
        e = q + a * s * (np.ptp(q) + 1e-9) * taper
    else:
        if op == "notch":
            ci = jn
        elif op == "late":
            ci = jn + OP_LATE_OFFSET * n
        elif op == "reflect":
            ci = i2 if i2 > i1 + 2 else i1 + 0.08 * n
        elif op == "upstroke":
            ci = OP_UPSTROKE_FRAC * i1
        else:
            raise ValueError(op)
        g = hann_kern(n, ci / max(n - 1, 1), OPS[op]["w"]) * taper
        e = q + OPS[op]["sign"] * a * g
    clipped = float(np.mean((e < 0) | (e > 1)))
    return np.clip(e, 0.0, 1.0), clipped


def l1_matrix(X):
    rows = []
    for x in X:
        f = pwfeat.seg_feats(x)
        rows.append([f.get(k, np.nan) for k in L1_FEATS])
    return np.asarray(rows, float)


def predict(model, Xarr, bs=2048):
    out = []
    with torch.no_grad():
        for i in range(0, len(Xarr), bs):
            xb = torch.as_tensor(Xarr[i:i + bs], dtype=torch.float32, device=P.DEV)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


# ------------------------------------------------------------------ S 特异度
def run_specificity(X, seed=0, n=None, label="PWDB"):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(X)) if n is None else rng.choice(
        len(X), min(n, len(X)), replace=False)
    Q = X[idx]
    V0 = l1_matrix(Q)
    sd = np.nanstd(V0, 0)
    sd = np.where(sd > 1e-12, sd, np.nan)
    rows, clip_rows, signed = [], [], {}
    for op, doses in DOSES.items():
        phase = rng.uniform(0, 2 * np.pi, 4) if op == "noise" else None
        for a in doses:
            Qe, cl = zip(*[apply_op(q, op, a, phase) for q in Q])
            Ve = l1_matrix(np.asarray(Qe))
            dz = (Ve - V0) / sd
            m = np.nanmean(np.abs(dz), 0)
            s = np.nanmean(dz, 0)
            for j, f in enumerate(L1_FEATS):
                if np.isfinite(m[j]):
                    rows.append(dict(source=label, op=op, dose=a, feat=f,
                                     cluster=F2C[f], family=F2FAM[f],
                                     mean_abs_dz=float(m[j]),
                                     signed_dz=float(s[j])))
            clip_rows.append(dict(source=label, op=op, dose=a,
                                  clip_frac=float(np.mean(cl)),
                                  n_cycle=len(Q)))
            if abs(a) == max(abs(np.array(doses))):
                signed[(op, a)] = s
    return (pd.DataFrame(rows), pd.DataFrame(clip_rows), signed, (V0, sd, idx))


def spec_summary(spec, label="PWDB"):
    """算子 → 簇/族的位移剖面。selectivity 只作诊断，不作主张（见模块 docstring）。"""
    out = []
    for op, g in spec.groupby("op"):
        a = float(g.dose.abs().max())
        gg = g[g.dose.abs() == a]
        per = gg.groupby("cluster").mean_abs_dz.max()
        fam = gg.groupby("family").mean_abs_dz.max()
        top = per.idxmax()
        second = float(per.drop(top).max()) if len(per) > 1 else np.nan
        out.append(dict(source=label, op=op, dominant_cluster=top,
                        prior_target=OPS[op]["target"],
                        prior_match=bool(top == OPS[op]["target"]),
                        effect_top=float(per.max()), effect_second=second,
                        selectivity=float(per.max() / max(second, 1e-9)),
                        fam_mechanism=float(fam.get("mechanism", np.nan)),
                        fam_descriptor=float(fam.get("descriptor", np.nan)),
                        fam_device=float(fam.get("device", np.nan)),
                        **{f"c_{c}": float(per.get(c, np.nan)) for c in CLUSTERS}))
    return pd.DataFrame(out)


def profile_similarity(spec):
    """算子之间的位移剖面余弦相似度：生理算子应彼此相似、与噪声算子分离。"""
    prof = (spec[spec.dose.abs() == spec.groupby("op").dose.transform(
        lambda s: s.abs().max())]
        .groupby(["op", "cluster"]).mean_abs_dz.max().unstack().fillna(0.0))
    V = prof.values
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    S = V @ V.T
    ops = list(prof.index)
    rows = [dict(op_i=ops[i], op_j=ops[j], cosine=float(S[i, j]))
            for i in range(len(ops)) for j in range(i + 1, len(ops))]
    return pd.DataFrame(rows), prof


def portability(signed_pw, signed_cm):
    """跨域可移植性：同一算子、同一批 L1 量，两个域的**有符号**位移是否同向。"""
    rows = []
    for key, vp in signed_pw.items():
        op, a = key
        if (op, a) not in signed_cm:
            continue
        vc = signed_cm[(op, a)]
        ok = np.isfinite(vp) & np.isfinite(vc)
        if ok.sum() < 5:
            continue
        rho = float(spearmanr(vp[ok], vc[ok]).statistic)
        rows.append(dict(op=op, dose=a, n_feat=int(ok.sum()), rho_pwdb_cm=rho,
                         same_sign_frac=float(np.mean(np.sign(vp[ok]) == np.sign(vc[ok])))))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ D 剂量-反应
def run_dose_response(model, X, cand=None, seed=0, n=800):
    rng = np.random.RandomState(seed + 7)
    pool = np.arange(len(X)) if cand is None else cand
    idx = rng.choice(pool, min(n, len(pool)), replace=False)
    Q = X[idx]
    base = predict(model, Q)
    rows = []
    for op, doses in DOSES.items():
        phase = rng.uniform(0, 2 * np.pi, 4) if op == "noise" else None
        for a in doses:
            Qe, cl = zip(*[apply_op(q, op, a, phase) for q in Q])
            p = predict(model, np.asarray(Qe))
            d = p - base
            bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(DOSE_BOOT)]
            rows.append(dict(op=op, dose=a, mean_dpred=float(d.mean()),
                             sd_dpred=float(d.std()),
                             lo=float(np.percentile(bs, 2.5)),
                             hi=float(np.percentile(bs, 97.5)),
                             clip_frac=float(np.mean(cl)),
                             frac_moved=float(np.mean(np.abs(d) > 0.01))))
    return pd.DataFrame(rows)


def monotonicity(dose_df):
    out = []
    for op, g in dose_df.groupby("op"):
        g = g.sort_values("dose")
        rho = float(spearmanr(g.dose, g.mean_dpred).statistic)
        out.append(dict(op=op, rho_dose=rho,
                        span=float(g.mean_dpred.max() - g.mean_dpred.min()),
                        sign_consistent=bool(abs(rho) > 0.9)))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ N 必要性
def run_necessity(R, cl_of_feat, ks=(1, 2, 3, 4, 6)):
    sae, st, model = R["sae"], R["st"], R["model"]
    Z = torch.as_tensor(R["Z"], dtype=torch.float32, device=P.DEV)
    with torch.no_grad():
        A_t = sae.encode((Z - st["mu"]) / st["sd"])
        base = model.head(sae.post(A_t) * st["sd"] + st["mu"]).squeeze(-1)
    rows = []
    for c in CLUSTERS:
        feats = [f for f in range(len(cl_of_feat)) if cl_of_feat[f] == c]
        if not feats:
            continue
        order = sorted(feats, key=lambda f: -R["d_pred"][f])
        grid = sorted({k for k in ks if k < len(order)} | {len(order)})
        for k in grid:
            with torch.no_grad():
                a2 = A_t.clone()
                a2[:, order[:k]] = 0.0
                p = model.head(sae.post(a2) * st["sd"] + st["mu"]).squeeze(-1)
                d = float((p - base).abs().mean())
            rows.append(dict(arch=R["arch"], cluster=c, k=k,
                             n_feat_in_cluster=len(order), frac_removed=k / len(order),
                             mean_abs_dpred=d))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ F 充分性
def _cv_pred(V, y, k=5, seed=0, cols=None):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    V = V if cols is None else V[:, cols]
    out = np.zeros(len(y), np.float32)
    for tr, te in KFold(k, shuffle=True, random_state=seed).split(V):
        out[te] = make_pipeline(StandardScaler(), Ridge(1.0)).fit(
            V[tr], y[tr]).predict(V[te])
    return out


def run_sufficiency(Vl1, pred_oof, seed=0, kmax=30):
    V = np.where(np.isfinite(Vl1), Vl1, np.nanmedian(Vl1, 0))
    keep = [j for j in range(V.shape[1]) if np.std(V[:, j]) > 1e-9]
    V = V[:, keep]
    names = [L1_FEATS[j] for j in keep]
    full = _cv_pred(V, pred_oof, seed=seed)
    r_full = float(spearmanr(full, pred_oof).statistic)
    r2_full = float(1 - ((full - pred_oof) ** 2).sum() /
                    ((pred_oof - pred_oof.mean()) ** 2).sum())
    cors = np.array([abs(spearmanr(V[:, j], pred_oof).statistic)
                     for j in range(V.shape[1])])
    order = list(np.argsort(-cors))
    par = []
    for k in range(1, min(kmax, len(order)) + 1):
        p = _cv_pred(V, pred_oof, seed=seed, cols=order[:k])
        par.append(dict(k=k, rho=float(spearmanr(p, pred_oof).statistic),
                        added=names[order[k - 1]]))
    par = pd.DataFrame(par)
    hit = par.rho >= 0.95 * r_full
    k95 = int(par.loc[hit, "k"].min()) if hit.any() else -1
    return dict(rho_full=r_full, r2_full=r2_full, n_feat=len(names), k95=k95), par


def run_surrogate_share(Vl1, pred_oof):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    V = np.where(np.isfinite(Vl1), Vl1, np.nanmedian(Vl1, 0))
    Vs = StandardScaler().fit_transform(V)
    beta = Ridge(1.0).fit(Vs, pred_oof).coef_
    share = {}
    for c in CLUSTERS:
        jj = [j for j, f in enumerate(L1_FEATS) if F2C[f] == c]
        share[c] = float(np.abs(beta[jj]).sum()) if jj else 0.0
    tot = sum(share.values()) + 1e-12
    return {c: v / tot for c, v in share.items()}


def run_convergence(a, b, d, names):
    cs = CLUSTERS
    v = [np.array([x.get(c, 0.0) for c in cs]) for x in (a, b, d)]
    rows = []
    for i, j, nm in [(0, 1, "SAE×干预"), (0, 2, "SAE×代理"), (1, 2, "干预×代理")]:
        rows.append(dict(pair=nm, rho=float(spearmanr(v[i], v[j]).statistic),
                         top1_agree=bool(np.argmax(v[i]) == np.argmax(v[j]))))
    tab = pd.DataFrame(dict(cluster=cs, sae_share=v[0], interv_share=v[1],
                            surro_share=v[2]))
    return pd.DataFrame(rows), tab


# ------------------------------------------------------------------ U 自助
def run_bootstrap(A, V0, d_pred, B=200, seed=0):
    """行自助：重算命名 → 簇份额区间。d_pred 固定（其抽样波动远小于命名）。"""
    rng = np.random.RandomState(seed)
    n = A.shape[0]
    assert V0.shape[0] == n, (V0.shape, A.shape)
    V = np.where(np.isfinite(V0), V0, np.nanmedian(V0, 0))
    recs = []
    for _ in range(B):
        ii = rng.randint(0, n, n)
        rho = P.assoc(A[ii], V[ii])
        cl = []
        for f in range(A.shape[1]):
            row = np.nan_to_num(rho[f], nan=0.0)
            if not np.isfinite(rho[f]).any() or (row ** 2).max() < 1e-12:
                cl.append("dead")
            else:
                cl.append(F2C[L1_FEATS[int(np.argmax(np.abs(row)))]])
        sh = {}
        for c in CLUSTERS:
            jj = [f for f in range(A.shape[1]) if cl[f] == c]
            sh[c] = float(d_pred[jj].sum()) if jj else 0.0
        tot = sum(sh.values()) + 1e-12
        recs.append({k: v / tot for k, v in sh.items()})
    df = pd.DataFrame(recs)
    return pd.DataFrame(dict(cluster=CLUSTERS, share_med=df.median().values,
                             lo=df.quantile(0.025).values,
                             hi=df.quantile(0.975).values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-dose", type=int, default=800)
    ap.add_argument("--arch", default="")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    n_spec = 500 if args.smoke else None
    epochs = 3 if args.smoke else P.MODEL_EPOCHS
    archs = [args.arch] if args.arch else list(P.ARCHS)
    sfx = f"_{args.tag}" if args.tag else ""

    X, T = P.load("rad")
    y = T["SBP_a"].values.astype(np.float32)
    y = (y - y.mean()) / y.std()
    print(f"[data] n={len(X)} dim={X.shape[1]} archs={archs} epochs={epochs}", flush=True)

    tcols = P.truth_cols(T)
    Vt = T[tcols].values.astype(np.float32)
    Vt = np.where(np.isfinite(Vt), Vt, np.nanmedian(Vt, 0))
    Vm, mcols = P.morph_matrix(X)
    Vl1 = l1_matrix(X)

    # ---- S（与模型无关）
    spec, clip, sp, _ = run_specificity(X, n=n_spec, label="PWDB")
    S = spec_summary(spec, "PWDB")
    sim, prof = profile_similarity(spec)
    print("\n[S] 生理算子应彼此相似、与噪声算子分离；descriptor 族必然同时移动")
    for _, r in S.iterrows():
        print(f"  {r.op:<9s} dominant={r.dominant_cluster:<11s}"
              f" prior={r.prior_target:<11s} match={'Y' if r.prior_match else 'N'}"
              f"  mech={r.fam_mechanism:.2f} desc={r.fam_descriptor:.2f}"
              f" dev={r.fam_device:.2f}")
    print("  剖面余弦（生理↔噪声 应最低）：")
    print("   " + "  ".join(f"{r.op_i}~{r.op_j}={r.cosine:.2f}"
                            for _, r in sim.iterrows()
                            if "noise" in (r.op_i, r.op_j)))

    # ---- 跨域可移植性
    zc = np.load(os.path.join(DATA, "real_cycles_v3.npz"), allow_pickle=True)
    Xc = zc["X"].astype(np.float32)
    Xc = (Xc - Xc.min(1, keepdims=True)) / (np.ptp(Xc, 1, keepdims=True) + 1e-9)
    cspec, cclip, csp, _ = run_specificity(Xc, label="CM")
    CS = spec_summary(cspec, "CM")
    port = portability(sp, csp)
    print("\n[S/跨域] 同一算子在 CM 真实周期上是否同向（有符号位移的秩相关）")
    for _, r in port.iterrows():
        print(f"  {r.op:<9s} dose={r.dose:+.2f} rho={r.rho_pwdb_cm:+.2f}"
              f" 同号={r.same_sign_frac:.0%}")

    all_nec, all_dose, all_mono, all_conv, all_suff, all_boot = [], [], [], [], [], []
    for arch in archs:
        R = P.run_arch(arch, X, y, Vt, tcols, Vm, mcols, epochs=epochs)
        A = R["A"]
        Vn = np.where(np.isfinite(Vl1), Vl1, np.nanmedian(Vl1, 0))
        rho_l1 = P.assoc(A, Vn)
        cl_of_feat = []
        for f in range(A.shape[1]):
            row = np.nan_to_num(rho_l1[f], nan=0.0)
            cl_of_feat.append("dead" if (row ** 2).max() < 1e-12
                              else F2C[L1_FEATS[int(np.argmax(np.abs(row)))]])
        tot = R["d_pred"].sum() + 1e-12
        sae_share = {c: float(R["d_pred"][[f for f in range(len(cl_of_feat))
                                           if cl_of_feat[f] == c]].sum() / tot)
                     for c in CLUSTERS}
        print(f"[{arch}] 每簇特征数 "
              + " ".join(f"{c}={Counter(cl_of_feat)[c]}" for c in CLUSTERS))

        nec = run_necessity(R, cl_of_feat)
        nec["arch"] = arch
        all_nec.append(nec)

        dose = run_dose_response(R["model"], X, n=args.n_dose)
        dose["arch"] = arch
        all_dose.append(dose)
        mono = monotonicity(dose)
        mono["arch"] = arch
        all_mono.append(mono)
        print(f"[{arch}] 剂量单调性 "
              + " ".join(f"{r.op}={r.rho_dose:+.2f}" for _, r in mono.iterrows()))
        mx = dose[dose.dose == dose.groupby("op").dose.transform("max")]
        print(f"[{arch}] 最大剂量 |Δpred| "
              + " ".join(f"{r.op}={abs(r.mean_dpred):.3f}" for _, r in mx.iterrows()))

        suff, par = run_sufficiency(Vl1, R["pred"])
        par["arch"] = arch
        par.to_csv(os.path.join(DATA, f"xinterp_parsimony_{arch}{sfx}.csv"),
                   index=False, encoding="utf-8-sig")
        all_suff.append(dict(arch=arch, **suff))
        print(f"[{arch}] 充分性 rho={suff['rho_full']:.3f} R2={suff['r2_full']:.3f}"
              f" k95={suff['k95']}")

        # C：三法 = 簇全消融 / 干预位移加权 / 代理系数
        full = nec.sort_values("k").groupby("cluster").tail(1).set_index("cluster")
        eff = {c: float(full.mean_abs_dpred.get(c, 0.0)) for c in CLUSTERS}
        te = sum(eff.values()) + 1e-12
        nec_share = {c: v / te for c, v in eff.items()}
        # 干预效应按各算子造成的簇位移比例分摊（软分摊，7 个簇都能拿到权重）
        Sx = spec_summary(spec, "PWDB").set_index("op")
        eff_i = {c: 0.0 for c in CLUSTERS}
        for op, r in Sx.iterrows():
            amp = float(dose[dose.op == op].mean_dpred.abs().max())
            w = np.array([r.get(f"c_{c}", 0.0) for c in CLUSTERS], float)
            w = np.nan_to_num(w, nan=0.0)
            w = w / (w.sum() + 1e-12)
            for i, c in enumerate(CLUSTERS):
                eff_i[c] += amp * w[i]
        ti = sum(eff_i.values()) + 1e-12
        interv_share = {c: v / ti for c, v in eff_i.items()}
        surro_share = run_surrogate_share(Vl1, R["pred"])
        conv, tab = run_convergence(nec_share, interv_share, surro_share, None)
        conv["arch"] = arch
        tab["arch"] = arch
        tab.to_csv(os.path.join(DATA, f"xinterp_clusters_{arch}{sfx}.csv"),
                   index=False, encoding="utf-8-sig")
        all_conv.append(conv)

        boot = run_bootstrap(A, Vl1, R["d_pred"], B=args.boot)
        boot["arch"] = arch
        all_boot.append(boot)

    out = lambda df, nm: df.to_csv(os.path.join(DATA, f"{nm}{sfx}.csv"), index=False,
                                   encoding="utf-8-sig")
    out(spec, "xinterp_specificity")
    out(clip, "xinterp_clip")
    out(cspec, "xinterp_cm_specificity")
    out(CS, "xinterp_spec_summary_cm")
    out(S, "xinterp_spec_summary")
    out(sim, "xinterp_profile_sim")
    out(port, "xinterp_portability")
    out(pd.concat(all_nec), "xinterp_necessity")
    out(pd.concat(all_dose), "xinterp_dose")
    out(pd.concat(all_mono), "xinterp_monotonicity")
    out(pd.concat(all_conv), "xinterp_convergence")
    out(pd.DataFrame(all_suff), "xinterp_sufficiency")
    out(pd.concat(all_boot), "xinterp_bootstrap")

    print("\n[C] 三法秩一致（允许失败，失败也是结果）")
    print(pd.concat(all_conv).to_string(index=False))
    print("\n[F] 充分性")
    print(pd.DataFrame(all_suff).to_string(index=False))
    print("\n[N] 整簇消融（k=全簇）")
    nn = pd.concat(all_nec)
    print(nn[nn.k == nn.n_feat_in_cluster].to_string(index=False))
    # 簇大小不同 → 全簇消融的 Δ 不可直接比。固定 k 才是同口径比较。
    kfix = min(3, int(nn.n_feat_in_cluster.min()))
    print(f"\n[N] 固定 k={kfix} 消融（同口径，排除簇大小混杂）")
    print(nn[nn.k == kfix].to_string(index=False))
    nn[nn.k == kfix].to_csv(os.path.join(DATA, f"xinterp_necessity_fixedk{kfix}{sfx}.csv"),
                            index=False, encoding="utf-8-sig")
    print(f"\n-> data/xinterp_*{sfx}.csv")


if __name__ == "__main__":
    main()
