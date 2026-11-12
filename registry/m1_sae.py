"""
M1: 在 PWDB 桡动脉周期波形上训练稀疏自编码器（SAE），并做三类评估
  1) 重构保真  R2
  2) 潜变量-机制对齐  ρ(z_k, AI_r / PWV_cf / SBP_a / age)
  3) 因果干预    固定其余潜变量、只沿 z_k 方向移动 → 解码波形形态是否按机制方向单调改变
输入统一做逐周期 min-max 归一化（= 未校准形态），与 E1 的 UNCAL 协议一致。
对照基线：同维 PCA + 线性探针。
"""

import os as _os, sys as _sys

# Resolved at run time: scripts/ holds sibling groups, the repository root holds data/.
_S = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _S)
for _d in _os.listdir(_S):
    if _d[:1].isdigit() and _os.path.isdir(_os.path.join(_S, _d)):
        _sys.path.append(_os.path.join(_S, _d))
import _repo  # noqa: E402

import numpy as np, pandas as pd, os, json, torch, torch.nn as nn, warnings
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = _repo.ROOT
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)

# ---------------- 数据 ----------------
z = np.load(os.path.join(ROOT, "data", "pwdb_cycles.npz"))
X = z["X_rad"].astype(np.float32)                       # (N,256) mmHg
T = pd.read_csv(os.path.join(ROOT, "data", "pwdb_truth.csv"))
keep_cols = ["AI_r", "PWV_cf", "SBP_a", "PP_a", "age"]
ok = (T.plausible.values == 1)
for c in keep_cols:
    ok &= np.isfinite(T[c].values)
idx = np.where(ok)[0]
X = X[idx]
# 逐周期 min-max → 未校准形态
X = (X - X.min(1, keepdims=True)) / (X.max(1, keepdims=True) - X.min(1, keepdims=True) + 1e-9)
Y = T.iloc[idx][keep_cols].reset_index(drop=True)
print(f"DATA n={len(X)}  dim={X.shape[1]}  device={DEV}")

Xt = torch.tensor(X, device=DEV)


class SAE(nn.Module):
    def __init__(self, d=256, h=128, k=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, k))
        self.dec = nn.Sequential(nn.Linear(k, h), nn.ReLU(), nn.Linear(h, d), nn.Sigmoid())

    def forward(self, x):
        zz = torch.relu(self.enc(x))
        return self.dec(zz), zz


def train_sae(lam, epochs=400, lr=3e-3):
    net = SAE().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        rec, zz = net(Xt)
        loss = ((rec - Xt) ** 2).mean() + lam * zz.abs().mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        rec, zz = net(Xt)
        r2 = 1 - ((rec - Xt) ** 2).sum().item() / ((Xt - Xt.mean(0)) ** 2).sum().item()
    return net, r2, zz.cpu().numpy(), rec.cpu().numpy()


print("\n[1/3] lambda 扫描（稀疏-保真权衡）")
sweep = []
best = None
for lam in [1e-4, 3e-4, 1e-3, 3e-3]:
    net, r2, Z, R = train_sae(lam)
    l0 = float((Z > 1e-6).sum(1).mean())
    alive = int((Z.std(0) > 1e-6).sum())
    sweep.append(dict(lam=lam, r2=round(r2, 4), l0=round(l0, 2), alive=alive))
    print(f"  lam={lam:<7} R2={r2:.4f}  L0={l0:.1f}/32  alive={alive}")
    if r2 >= 0.93 and (best is None or l0 < best[1]):
        best = (lam, l0, net, r2, Z, R)
pd.DataFrame(sweep).to_csv(os.path.join(ROOT, "data", "m1_sae_sweep.csv"), index=False)
assert best is not None, "无配置满足 R2>=0.93"
lam, l0, net, r2, Z, R = best
print(f"  -> 选用 lam={lam}  R2={r2:.4f}  L0={l0:.1f}")

# ---------------- 对齐 ----------------
print("\n[2/3] 潜变量-机制对齐（Spearman）")
rows = []
for k in range(Z.shape[1]):
    if Z[:, k].std() < 1e-6:
        continue
    d = {"latent": k, "active_frac": round(float((Z[:, k] > 1e-6).mean()), 3)}
    for c in keep_cols:
        d["rho_" + c] = round(float(spearmanr(Z[:, k], Y[c])[0]), 3)
    rows.append(d)
A = pd.DataFrame(rows)
A["absmax"] = A[["rho_" + c for c in keep_cols]].abs().max(1)
A = A.sort_values("absmax", ascending=False).reset_index(drop=True)
A.to_csv(os.path.join(ROOT, "data", "m1_latent_alignment.csv"), index=False)
print(A.head(8).to_string(index=False))

# ---------------- 探针：SAE 潜变量 vs PCA 基线 ----------------
def cv_probe(Fm, y, k=5, seed=0):
    ii = np.random.RandomState(seed).permutation(len(y))
    out = np.full(len(y), np.nan)
    for te in np.array_split(ii, k):
        tr = np.setdiff1d(ii, te)
        m = make_pipeline(StandardScaler(), Ridge(1.0)).fit(Fm[tr], y[tr])
        out[te] = m.predict(Fm[te])
    return out

P32 = PCA(32, random_state=0).fit_transform(X)
print("\n[3/3] 探针对比  ρ（5 折）")
cmp_rows = []
for c in keep_cols:
    y = Y[c].values.astype(float)
    r_sae = spearmanr(cv_probe(Z, y), y)[0]
    r_pca = spearmanr(cv_probe(P32, y), y)[0]
    r_pix = spearmanr(cv_probe(X, y), y)[0]
    cmp_rows.append(dict(target=c, rho_sae32=round(float(r_sae), 3),
                         rho_pca32=round(float(r_pca), 3), rho_rawpix=round(float(r_pix), 3)))
C = pd.DataFrame(cmp_rows)
C.to_csv(os.path.join(ROOT, "data", "m1_probe_compare.csv"), index=False)
print(C.to_string(index=False))

# ---------------- 因果干预 ----------------
print("\n[4/4] 因果干预：固定其余潜变量，只移动单个潜变量")
target_lat = int(A.loc[A.rho_PWV_cf.abs().idxmax(), "latent"])
rnd_lat = int(A.loc[A.rho_PWV_cf.abs().nsmallest(3).index[0], "latent"])
print(f"  目标潜变量 z{target_lat}（|rho PWV| 最大）  对照 z{rnd_lat}（|rho| 最小）")


def aix_from_wave(p):
    q = (p - p.min()) / (p.max() - p.min() + 1e-9)
    n = len(q); i1 = int(np.argmax(q[: int(.55 * n)]))
    tail = q[i1:]
    cand = [i for i in range(1, len(tail) - 1) if tail[i] >= tail[i - 1] and tail[i] >= tail[i + 1]]
    i2 = i1 + cand[int(np.argmax([tail[i] for i in cand]))] if cand else i1
    Fq = np.abs(np.fft.rfft(q - q.mean()))
    return float(q[i2] - q[i1]), float(Fq[2] / (Fq[1] + 1e-9))


with torch.no_grad():
    Zt = torch.tensor(Z, device=DEV)
    base_rec = net.dec(Zt).cpu().numpy()
sd = Z[:, target_lat].std() + 1e-9
sweep_rows = []
grid = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
for dz in grid:
    Zmod = Z.copy(); Zmod[:, target_lat] = Z[:, target_lat] + dz * sd
    with torch.no_grad():
        w = net.dec(torch.tensor(Zmod, device=DEV)).cpu().numpy()
        # 流形校验：解码出的波形再编码，是否回到被设定的潜变量
        z_back = torch.relu(net.enc(torch.tensor(w, device=DEV))).cpu().numpy()
    manifold_err = float(np.abs(z_back - Zmod).mean())
    ap = np.array([aix_from_wave(x) for x in w])
    sweep_rows.append(dict(dz=dz, aix_mean=round(float(ap[:, 0].mean()), 4),
                           h2_mean=round(float(ap[:, 1].mean()), 4),
                           reenc_err=round(manifold_err, 4)))
S = pd.DataFrame(sweep_rows)
print(S.to_string(index=False))
S.to_csv(os.path.join(ROOT, "data", "m1_intervention.csv"), index=False)

rho_int = spearmanr(S.dz, S.aix_mean)[0]
viol = int(np.sum(np.diff(S.aix_mean.values) * np.sign(S.aix_mean.iloc[-1] - S.aix_mean.iloc[0]) < 0))
print(f"  干预曲线 rho(dz, AIx) = {rho_int:+.3f}  方向反转次数 = {viol}/{len(S)-1}")
print(f"  AIx 变化量 dz=-2→+2: {S.aix_mean.iloc[-1] - S.aix_mean.iloc[0]:+.4f}")
print(f"  再编码平均误差 = {S.reenc_err.mean():.4f}（0=完全落回流形）")

# 靶向消融：抹掉某个潜变量后，各读出损失多少
print("\n  靶向消融（把 z_k 置 0，看读出损失）")
abl_rows = []
for c in keep_cols:
    y = Y[c].values.astype(float)
    base = spearmanr(cv_probe(Z, y), y)[0]
    row = dict(target=c, full=round(float(base), 3))
    for k in [target_lat, rnd_lat]:
        Zk = Z.copy(); Zk[:, k] = 0.0
        row[f"drop_z{k}"] = round(float(spearmanr(cv_probe(Zk, y), y)[0]), 3)
    abl_rows.append(row)
AB = pd.DataFrame(abl_rows)
AB.to_csv(os.path.join(ROOT, "data", "m1_ablation.csv"), index=False)
print(AB.to_string(index=False))

np.savez(os.path.join(ROOT, "data", "m1_sae.npz"),
         Z=Z, rec=R, X=X, latent_target=target_lat, latent_ctrl=rnd_lat,
         W_dec=net.dec[0].weight.detach().cpu().numpy(),
         W_enc=net.enc[2].weight.detach().cpu().numpy(), lam=lam, r2=r2)
print("\nsaved m1_sae.npz / m1_latent_alignment.csv / m1_probe_compare.csv / m1_intervention.csv")
