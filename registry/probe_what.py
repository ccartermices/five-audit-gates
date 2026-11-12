"""拆开一个做任务的模型，看它到底依据了什么。支持多架构对照。

与 m1_sae.py 的分工
--------------------
m1 解剖的是**自编码器**：波形 -> 潜变量 -> 还原波形。回答"波形由哪几个因子组成"。
本文件解剖的是**一个做判断的模型**：波形 -> 隐表征 -> 血压预测。回答"它凭什么这么判断"。

为什么能判断"想得对不对"
------------------------
PWDB 是仿真数据，每个波形的生成参数全部记录在 pwdb_truth.csv（129 列）。
不必依赖专家标注，可以直接问：这个 SAE 特征对应哪个真实物理量。
对应血管硬度/波反射 -> 想对了；主要对应 age 而该特征又驱动预测 -> 走捷径。

为什么按机制簇而不是按列名归因
------------------------------
物理量之间高度共线（PP_amp 与 MBP 的 |rho| 常在 0.9 以上，PWV 各段之间同理），
要求"一个特征只被一个列名解释"几乎不可能满足。所以归因到机制簇，并另报簇内份额。
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


import csv
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
# --- 可复现性 --------------------------------------------------------------
# 同一 seed 两次运行曾被观测到 Transformer 支路的 SAE 字典重排（d_pred 相关仅 0.66），
# 根因是 CUDA 上的非确定内核。种子研究的全部意义在于「只有 seed 变」，
# 所以这里强制确定性；CUBLAS_WORKSPACE_CONFIG 必须在 CUDA 初始化前设置。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:  # 老版本 torch 没有该接口
    pass
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pwfeat  # noqa: E402

ROOT = _repo.ROOT
DATA = os.path.join(ROOT, "data")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

SKIP = ("idx", "age_cfg", "plausible", "var_")

# 方法学参数提升为模块级常量：论文 Methods 里的每个数都要能被登记表按名字取走，
# 藏在函数默认值里就无法溯源（改了也没人知道）。
K_FOLD = 5           # 交叉验证折数
MODEL_EPOCHS = 80    # 任务模型训练轮数
SAE_N_FEAT = 128     # TopK-SAE 字典宽度
SAE_K_DEFAULT = 8    # TopK-SAE 的 k（基准配置）
SAE_EPOCHS = 3000    # SAE 训练轮数

# 机制簇：回答"它依据的是血管硬度、压力幅度、还是年龄"
# 分两类看：VASCULAR 是真正的机制解释，PRESSURE 是血压本身的各种表现（不是独立机制），
# DEMOGRAPHIC 是捷径，FLOW 是心脏泵血。判断"想得对不对"看前两者之比。
MECH = (
    ("stiffness", ("PWV", "AI_", "AP_", "AGI", "Tr_", "RI", "SI", "dPdt"),
     "vascular"),
    ("resistance", ("svr", "pvr", "pvc"), "vascular"),
    ("compliance", ("ac", "tau"), "vascular"),
    ("pressure", ("SBP", "DBP", "MBP", "PP_", "Pst_", "P1pk", "P2pk",
                  "P1in", "P2in", "Ps_"), "pressure"),
    ("geometry", ("dia_", "len_", "IAD"), "vascular"),
    ("demographic", ("age",), "demographic"),
    ("flow", ("SV", "CO", "HR", "LVET", "PFT", "RFV"), "flow"),
)

# 单列名精确匹配，避免 "c" 用前缀匹配时误吞其它列
EXACT = {"c": "stiffness"}


def mech_of(name):
    if name in EXACT:
        return EXACT[name]
    for tag, pre, _ in MECH:
        if any(name.startswith(x) for x in pre):
            return tag
    return "other"


def family_of(tag):
    for t, _, fam in MECH:
        if t == tag:
            return fam
    return "other"


def _family_agg(mech_causal):
    agg = Counter()
    for t, v in mech_causal.items():
        agg[family_of(t)] += v
    return dict(agg)


# ---------------------------------------------------------------- 数据
def load(which="rad", normalise=True):
    z = np.load(os.path.join(DATA, "pwdb_cycles.npz"), allow_pickle=True)
    X = z[f"X_{which}"].astype(np.float32)
    T = pd.read_csv(os.path.join(DATA, "pwdb_truth.csv"))
    ok = (T.plausible.values == 1)
    for c in ("SBP_a", "AI_r", "PWV_cf"):
        ok &= np.isfinite(T[c].values)
    idx = np.where(ok)[0]
    X, T = X[idx], T.iloc[idx].reset_index(drop=True)
    if normalise:
        # 逐周期 min-max：与 06/07 的 UNCAL 协议一致，也是不同设备唯一可比的口径
        X = (X - X.min(1, keepdims=True)) / (X.max(1, keepdims=True)
                                             - X.min(1, keepdims=True) + 1e-9)
    return X, T


def truth_cols(T):
    cols = []
    for c in T.columns:
        if any(c.startswith(s) for s in SKIP):
            continue
        v = T[c].values
        if v.dtype.kind not in "if":
            continue
        if np.isfinite(v).mean() < 0.9 or np.nanstd(v) < 1e-9:
            continue
        cols.append(c)
    return cols


def morph_matrix(X):
    feats = [pwfeat.seg_feats(x) for x in X]
    names = list(feats[0].keys())
    V = np.array([[f.get(k, np.nan) for k in names] for f in feats], np.float32)
    V = np.where(np.isfinite(V), V, np.nanmedian(V, 0))
    keep = [j for j in range(V.shape[1]) if np.std(V[:, j]) > 1e-9]
    return V[:, keep], [names[j] for j in keep]


# ---------------------------------------------------------------- 三种架构
class CNN(nn.Module):
    def __init__(self, h=64, p=0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 5, padding=2), nn.GELU(), nn.AdaptiveAvgPool1d(1))
        self.proj = nn.Sequential(nn.Linear(64, h), nn.GELU(), nn.Dropout(p))
        self.head = nn.Linear(h, 1)

    def encode(self, x):
        return self.proj(self.body(x.unsqueeze(1)).squeeze(-1))

    def forward(self, x):
        return self.head(self.encode(x)).squeeze(-1)


class MLP(nn.Module):
    def __init__(self, h=64, p=0.1):
        super().__init__()
        self.body = nn.Sequential(nn.Flatten(), nn.Linear(256, 128), nn.GELU(),
                                  nn.Dropout(p), nn.Linear(128, 128), nn.GELU())
        self.proj = nn.Sequential(nn.Linear(128, h), nn.GELU(), nn.Dropout(p))
        self.head = nn.Linear(h, 1)

    def encode(self, x):
        return self.proj(self.body(x))

    def forward(self, x):
        return self.head(self.encode(x)).squeeze(-1)


class Former(nn.Module):
    """把 256 点切成 31 个 token，走 2 层自注意力。"""

    def __init__(self, h=64, d=32, n_layer=2, n_head=4, p=0.1):
        super().__init__()
        self.patch = nn.Conv1d(1, d, 16, stride=8)
        self.pos = nn.Parameter(torch.zeros(1, 64, d))
        layer = nn.TransformerEncoderLayer(d, n_head, d * 2, p, batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layer)
        self.proj = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(p))
        self.head = nn.Linear(h, 1)

    def encode(self, x):
        t = self.patch(x.unsqueeze(1)).transpose(1, 2)      # (B, L, d)
        t = t + self.pos[:, :t.shape[1]]
        return self.proj(self.enc(t).mean(1))

    def forward(self, x):
        return self.head(self.encode(x)).squeeze(-1)


ARCHS = {"cnn": CNN, "mlp": MLP, "transformer": Former}


class TopKSAE(nn.Module):
    def __init__(self, d_in, n_feat=128, k=8):
        super().__init__()
        self.k, self.n_feat = k, n_feat
        self.pre = nn.Linear(d_in, n_feat)
        self.post = nn.Linear(n_feat, d_in, bias=False)

    def encode(self, z):
        a = torch.relu(self.pre(z))
        if self.k < self.n_feat:
            thr = a.topk(self.k, 1).values[:, -1:]
            a = a * (a >= thr)
        return a

    def forward(self, z):
        a = self.encode(z)
        return self.post(a), a


def fit_model(arch, Xtr, ytr, Xte=None, epochs=MODEL_EPOCHS, seed=0, h=64):
    torch.manual_seed(seed)
    m = ARCHS[arch](h).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    xa = torch.tensor(Xtr, device=DEV)
    ya = torch.tensor(ytr, device=DEV)
    n, bs = len(Xtr), 256
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            opt.zero_grad()
            ((m(xa[b]) - ya[b]) ** 2).mean().backward()
            opt.step()
    m.eval()
    out = {}
    with torch.no_grad():
        if Xte is not None:
            xt = torch.tensor(Xte, device=DEV)
            out["pred"] = m(xt).cpu().numpy()
            out["Z"] = m.encode(xt).cpu().numpy()
    return m, out


def oof(arch, X, y, k=K_FOLD, seed=0, epochs=MODEL_EPOCHS):
    rs = np.random.RandomState(seed)
    ii = rs.permutation(len(y))
    pred = np.zeros(len(y), np.float32)
    for te in np.array_split(ii, k):
        tr = np.setdiff1d(ii, te)
        _, o = fit_model(arch, X[tr], y[tr], X[te], epochs=epochs, seed=seed)
        pred[te] = o["pred"]
    return pred


def train_sae(Z, n_feat=SAE_N_FEAT, k=SAE_K_DEFAULT, epochs=SAE_EPOCHS, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    mu = Z.mean(0, keepdim=True)
    sd = Z.std(0, keepdim=True).clamp(min=1e-6)
    Zn = (Z - mu) / sd
    m = TopKSAE(Z.shape[1], n_feat, k).to(Z.device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        rec, _ = m(Zn)
        ((rec - Zn) ** 2).mean().backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        rec, a = m(Zn)
        r2 = 1 - ((rec - Zn) ** 2).sum().item() / ((Zn - Zn.mean(0)) ** 2).sum().item()
    return m, dict(r2=float(r2), l0=float((a > 0).float().sum(1).mean()),
                   dead=int(((a > 0).sum(0) == 0).sum()), mu=mu, sd=sd)


def assoc(A, V):
    """向量化 Spearman：先转秩再求相关矩阵。A:(n,F) V:(n,d) -> (F,d)"""
    def rz(M):
        R = np.apply_along_axis(rankdata, 0, M)
        R = R - R.mean(0)
        s = R.std(0)
        return R / np.where(s > 1e-12, s, 1.0)
    C = (rz(A).T @ rz(V)) / len(A)
    C[~(A.std(0) > 1e-9)] = np.nan
    return C


def attribute(rho, names, thr=0.5):
    """每特征：最强的物理量、强度、所属机制簇、簇内份额、是否可命名。

    簇内份额 = 同簇 rho^2 之和 / 全部 rho^2 之和。用它而不是单列纯度，
    因为物理量彼此共线，同簇多列同时强相关是正常现象而非缺陷。
    """
    out = []
    tags = [mech_of(n) for n in names]
    for f in range(rho.shape[0]):
        row = np.nan_to_num(rho[f], nan=0.0)
        pw = row ** 2
        if not np.isfinite(rho[f]).any() or pw.max() < 1e-12:
            out.append(dict(top="", rho=np.nan, mech="dead", share=np.nan,
                            named="no"))
            continue
        j = int(np.argmax(np.abs(row)))
        m = tags[j]
        share = pw[[i for i, t in enumerate(tags) if t == m]].sum() / max(pw.sum(), 1e-12)
        out.append(dict(top=names[j], rho=float(row[j]), mech=m,
                        share=float(share),
                        named="yes" if abs(row[j]) >= thr else "no"))
    return out


def run_arch(arch, X, y, Vt, tcols, Vm, mcols, epochs=MODEL_EPOCHS, seed=0, sae_k=SAE_K_DEFAULT):
    print(f"\n=== {arch} ===", flush=True)
    pred = oof(arch, X, y, k=5, seed=seed, epochs=epochs)
    rho_task = float(spearmanr(pred, y).statistic)
    r2_task = float(1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    model, out = fit_model(arch, X, y, X, epochs=epochs, seed=seed)
    Z = out["Z"]
    npar = sum(p.numel() for p in model.parameters())
    print(f"[{arch}] OOF rho={rho_task:.3f} R2={r2_task:.3f} params={npar:,}", flush=True)

    Zt = torch.tensor(Z, device=DEV)
    sae, st = train_sae(Zt, k=sae_k, seed=seed)
    with torch.no_grad():
        A_t = sae.encode((Zt - st["mu"]) / st["sd"])
        A = A_t.cpu().numpy()
        base_p = model.head(sae.post(A_t) * st["sd"] + st["mu"]).squeeze(-1)
    print(f"[{arch}] SAE R2={st['r2']:.4f} 活特征={sae.n_feat - st['dead']}/{sae.n_feat}",
          flush=True)

    rho_t = assoc(A, Vt)
    nm_t = attribute(rho_t, tcols)
    rho_m = assoc(A, Vm)
    nm_m = attribute(rho_m, mcols)

    # 因果：置零特征，走完整 Z -> a -> Z' -> 头
    d_pred = np.zeros(sae.n_feat, np.float32)
    for j in range(sae.n_feat):
        with torch.no_grad():
            a2 = A_t.clone()
            a2[:, j] = 0.0
            z2 = sae.post(a2) * st["sd"] + st["mu"]
            d_pred[j] = float((model.head(z2).squeeze(-1) - base_p).abs().mean())

    alive = [i for i in range(sae.n_feat) if nm_t[i]["mech"] != "dead"]
    cm_all = Counter(nm_t[i]["mech"] for i in alive)
    # 因果加权：按 Δ预测 把机制簇的贡献度分摊
    tot = d_pred.sum() + 1e-12
    cm_causal = {t: float(d_pred[[i for i in alive if nm_t[i]["mech"] == t]].sum() / tot)
                 for t in cm_all}
    named = sum(1 for i in alive if nm_t[i]["named"] == "yes")

    print(f"[{arch}] 可命名 {named}/{len(alive)} 个活特征；机制簇（按因果贡献）:")
    for t, v in sorted(cm_causal.items(), key=lambda x: -x[1]):
        print(f"    {t:<12s} 个数 {cm_all[t]:>3d}  因果份额 {100*v:>5.1f}%")

    rows = []
    for f in range(sae.n_feat):
        rows.append(dict(arch=arch, feat=f,
                         act_mean=round(float(A[:, f].mean()), 5),
                         mech=nm_t[f]["mech"], top_phys=nm_t[f]["top"],
                         rho_phys="" if not np.isfinite(nm_t[f]["rho"])
                         else round(nm_t[f]["rho"], 3),
                         mech_share="" if not np.isfinite(nm_t[f]["share"])
                         else round(nm_t[f]["share"], 3),
                         named=nm_t[f]["named"],
                         top_morph=nm_m[f]["top"],
                         rho_morph="" if not np.isfinite(nm_m[f]["rho"])
                         else round(nm_m[f]["rho"], 3),
                         d_pred=round(float(d_pred[f]), 5)))
    # model / sae / st 也返回：下游的干预审计要在同一权重上做反事实推理，
    # 重新训练一遍就会换一个模型，干预结果无法与归因结果对齐。
    return dict(arch=arch, rho_task=rho_task, r2_task=r2_task, npar=npar,
                sae_r2=st["r2"], alive=len(alive), named=named,
                mech_count=dict(cm_all), mech_causal=cm_causal,
                fam_causal=_family_agg(cm_causal), rows=rows,
                A=A, Z=Z, rho_t=rho_t, rho_m=rho_m, d_pred=d_pred,
                pred=pred, tcols=tcols, mcols=mcols,
                model=model, sae=sae, st=st)


def main(seed=0, sae_k=SAE_K_DEFAULT, tag=""):
    # 种子稳定性 / k 扫描：默认值 (0, 8, "") 与最初那版完全一致，原结果可原位复现；
    # 带 tag 时产物加后缀，绝不覆盖已发表的基线产物。
    sfx = f"_{tag}" if tag else ""
    print(f"[env] device={DEV}  seed={seed}  sae_k={sae_k}  tag={tag or '-'}")
    X, T = load("rad")
    y = T["SBP_a"].values.astype(np.float32)
    y = (y - y.mean()) / y.std()
    print(f"[data] n={len(X)} dim={X.shape[1]}  SBP_a={T.SBP_a.min():.0f}-"
          f"{T.SBP_a.max():.0f} mmHg")

    tcols = truth_cols(T)
    Vt = T[tcols].values.astype(np.float32)
    Vt = np.where(np.isfinite(Vt), Vt, np.nanmedian(Vt, 0))
    Vm, mcols = morph_matrix(X)

    # 同口径上限：只用波形形态量能做到多少
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    hm = np.zeros(len(y), np.float32)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(Vm):
        hm[te] = make_pipeline(StandardScaler(), Ridge(1.0)).fit(
            Vm[tr], y[tr]).predict(Vm[te])
    morph_ceil = float(spearmanr(hm, y).statistic)
    print(f"[ceiling] 仅用 {len(mcols)} 个形态量 -> rho={morph_ceil:.3f}")

    # 捷径基线：只给年龄，能预测到多少。不给这个数，就说不清"用年龄"算不算走捷径
    age = T["age"].values.astype(np.float32).reshape(-1, 1)
    ha = np.zeros(len(y), np.float32)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(age):
        ha[te] = make_pipeline(StandardScaler(), Ridge(1.0)).fit(
            age[tr], y[tr]).predict(age[te])
    age_ceil = float(spearmanr(ha, y).statistic)
    print(f"[ceiling] 仅用年龄 -> rho={age_ceil:.3f}（捷径基线）")

    res = [run_arch(a, X, y, Vt, tcols, Vm, mcols, seed=seed, sae_k=sae_k)
           for a in ARCHS]

    rows = [r for R in res for r in R["rows"]]
    fn = os.path.join(DATA, f"probe_what{sfx}.csv")
    with open(fn, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez(os.path.join(DATA, f"probe_what{sfx}.npz"),
             **{f"{R['arch']}_A": R["A"] for R in res},
             **{f"{R['arch']}_Z": R["Z"] for R in res},
             **{f"{R['arch']}_rho_t": R["rho_t"] for R in res},
             **{f"{R['arch']}_dpred": R["d_pred"] for R in res},
             **{f"{R['arch']}_pred": R["pred"] for R in res},
             **{f"{R['arch']}_fam": np.array(
                 [R["fam_causal"].get(f, 0.0) for f in
                  ["vascular", "pressure", "flow", "demographic", "other"]])
                for R in res},
             y=y, tcols=np.array(tcols), mcols=np.array(mcols),
             morph_ceiling=morph_ceil)

    print("\n" + "=" * 68)
    print(f"{'arch':<12s}{'rho':>7s}{'R2':>7s}{'params':>10s}{'alive':>7s}"
          f"{'named':>7s}{'SAE R2':>9s}")
    for R in res:
        print(f"{R['arch']:<12s}{R['rho_task']:>7.3f}{R['r2_task']:>7.3f}"
              f"{R['npar']:>10,d}{R['alive']:>7d}{R['named']:>7d}"
              f"{R['sae_r2']:>9.4f}")
    print(f"{'形态量上限':<12s}{morph_ceil:>7.3f}")

    print("\n机制簇的因果份额（Δ预测占比）")
    mechs = [t for t, _, _ in MECH] + ["other"]
    print(f"{'arch':<12s}" + "".join(f"{m:>13s}" for m in mechs))
    for R in res:
        print(f"{R['arch']:<12s}"
              + "".join(f"{100*R['mech_causal'].get(m, 0):>12.1f}%" for m in mechs))

    # 家族层面：血管力学 vs 压力表现 vs 年龄
    fams = ["vascular", "pressure", "flow", "demographic", "other"]
    print("\n机制家族（血管力学 = 真解释；压力 = 目标自身的表现；年龄 = 捷径）")
    print(f"{'arch':<12s}" + "".join(f"{f:>13s}" for f in fams))
    for R in res:
        agg = R["fam_causal"]
        print(f"{R['arch']:<12s}"
              + "".join(f"{100*agg.get(f, 0):>12.1f}%" for f in fams))

    meta = [("key", "value"), ("morph_ceiling", round(morph_ceil, 4)),
            ("age_ceiling", round(age_ceil, 4)),
            ("n", len(y)), ("n_truth", len(tcols)), ("n_morph", len(mcols))]
    for R in res:
        meta += [(f"{R['arch']}_rho", round(R["rho_task"], 4)),
                 (f"{R['arch']}_params", R["npar"]),
                 (f"{R['arch']}_alive", R["alive"]),
                 (f"{R['arch']}_named", R["named"])]
    with open(os.path.join(DATA, f"probe_what_meta{sfx}.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        csv.writer(fh).writerows(meta)
    print(f"\n-> {fn}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="SAE 解剖任务模型（种子/k 可扫）")
    ap.add_argument("--seed", type=int, default=0, help="模型与 SAE 的训练种子")
    ap.add_argument("--sae-k", type=int, default=SAE_K_DEFAULT, help="TopK-SAE 的 k")
    ap.add_argument("--tag", default="", help="产物后缀（空=原位覆盖基线）")
    a = ap.parse_args()
    main(seed=a.seed, sae_k=a.sae_k, tag=a.tag)
