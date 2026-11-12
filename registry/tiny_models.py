"""
极小模型 vs 现有 SAE vs JEPA 式联合嵌入：单周期脉搏波的容量曲线。

要回答四个问题：
  Q1 内在维度只有 2-3.5，把模型压到几千参数会不会掉性能？
  Q2 决定性能的是"模型宽度"还是"潜变量维度"？
  Q3 预测式目标（JEPA/BYOL 式联合嵌入）能否替代重构式目标（SAE）？
  Q4 统一语料（34 维形态特征）上，2-3 维线性瓶颈是否够支持跨域 SBP 回归？

关于 JEPA：四个数据源每个受试者只有一条"代表周期"，没有时间序，
无法做经典 JEPA 的"预测下一时刻潜变量"。这里退到非对比式联合嵌入：
增强视图 v1 -> online encoder -> predictor 预测 EMA target encoder(v2)，
再加 VICReg 方差/协方差项防塌缩（BYOL 骨架，属 JEPA 家族）。

训练配方说明：纯 ReLU 潜变量在小瓶颈下会整层死亡（见 sae_collapse 行），
因此瓶颈 AE 用 GELU + 线性潜变量 + 2000 epoch，并把 3 个种子的中位数作为结果。
"""

import os as _os, sys as _sys

# Resolved at run time: scripts/ holds sibling groups, the repository root holds data/.
_S = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _S)
for _d in _os.listdir(_S):
    if _d[:1].isdigit() and _os.path.isdir(_os.path.join(_S, _d)):
        _sys.path.append(_os.path.join(_S, _d))
import _repo  # noqa: E402

import os, sys, csv, json, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
ROOT = _repo.ROOT
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, os.path.join(ROOT, "scripts"))

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 20261031
SEEDS = [0, 1, 2]
torch.manual_seed(SEED)
np.random.seed(SEED)


# ------------------------------------------------------------------ 数据
def load_pwdb():
    z = np.load(os.path.join(DATA, "pwdb_cycles.npz"))
    X = z["X_rad"].astype(np.float32)
    rows = list(csv.DictReader(open(os.path.join(DATA, "pwdb_truth.csv"),
                                    encoding="utf-8-sig")))
    cols = ["AI_r", "PWV_cf", "SBP_a", "PP_a", "age"]
    ok = np.array([int(float(r["plausible"])) == 1 for r in rows])
    Y = {c: np.array([float(r[c]) if r[c] not in ("", "nan") else np.nan
                      for r in rows]) for c in cols}
    for c in cols:
        ok &= np.isfinite(Y[c])
    X = X[ok]
    Y = {c: v[ok] for c, v in Y.items()}
    # 逐周期 min-max = 未校准形态（与 M1 协议一致）
    X = (X - X.min(1, keepdims=True)) / (X.max(1, keepdims=True) -
                                         X.min(1, keepdims=True) + 1e-9)
    return X, Y


def resample(X, R):
    """把 (N,256) 周期按相位线性重采样到 R 点。"""
    n = X.shape[1]
    src = np.linspace(0, 1, n)
    dst = np.linspace(0, 1, R)
    out = np.empty((len(X), R), np.float32)
    for i in range(len(X)):
        out[i] = np.interp(dst, src, X[i])
    return out


def nparam(m):
    return int(sum(p.numel() for p in m.parameters()))


def effrank(Z):
    """参与率 PR = (Σλ)² / Σλ²，对 Z 的协方差做。"""
    Z = np.asarray(Z, float)
    Z = Z - Z.mean(0)
    C = Z.T @ Z / max(len(Z) - 1, 1)
    lam = np.clip(np.linalg.eigvalsh(C), 0, None)
    return float(lam.sum() ** 2 / (np.sum(lam ** 2) + 1e-12))


# ------------------------------------------------------------------ 读出
def probe(F, y, k=5, seed=SEED):
    """5 折 OOF 预测后用 Spearman 评价，避免折内高估。"""
    ii = np.random.RandomState(seed).permutation(len(y))
    oof = np.full(len(y), np.nan)
    for te in np.array_split(ii, k):
        tr = np.setdiff1d(ii, te)
        m = make_pipeline(StandardScaler(), Ridge(1.0)).fit(F[tr], y[tr])
        oof[te] = m.predict(F[te])
    return float(spearmanr(oof, y)[0])


# ------------------------------------------------------------------ 模型
class AE(nn.Module):
    """瓶颈自编码器。sparse=False 时潜变量为线性（等价非线性 PCA）。"""

    def __init__(self, d=256, h=32, k=3, sparse=False):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, k))
        self.dec = nn.Sequential(nn.Linear(k, h), nn.GELU(), nn.Linear(h, d),
                                 nn.Sigmoid())
        self.sparse = sparse

    def forward(self, x):
        z = self.enc(x)
        if self.sparse:
            z = torch.relu(z)
        return self.dec(z), z


def train_ae(X, h, k, lam=0.0, epochs=2000, lr=3e-3, seed=0, sparse=False):
    torch.manual_seed(seed)
    Xt = torch.tensor(X, device=DEV)
    net = AE(X.shape[1], h, k, sparse).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        rec, z = net(Xt)
        loss = ((rec - Xt) ** 2).mean() + lam * z.abs().mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        rec, z = net(Xt)
        r2 = 1 - ((rec - Xt) ** 2).sum().item() / ((Xt - Xt.mean(0)) ** 2).sum().item()
    return net, float(r2), z.cpu().numpy()


class Enc(nn.Module):
    def __init__(self, d=256, h=32, k=3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, k))

    def forward(self, x):
        return self.net(x)


class Pred(nn.Module):
    def __init__(self, h=32, k=3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(k, h), nn.GELU(), nn.Linear(h, k))

    def forward(self, z):
        return self.net(z)


def augment(x, g):
    """物理合理的周期波形增强：循环时移 / 幅值缩放 / 轻度平滑 / 噪声。"""
    B, d = x.shape
    r = torch.randint(-8, 9, (B, 1), generator=g, device=x.device)
    idx = (torch.arange(d, device=x.device).unsqueeze(0) + r) % d
    y = torch.gather(x, 1, idx)
    y = y * (0.9 + 0.2 * torch.rand(B, 1, generator=g, device=x.device))
    sm = (torch.roll(y, 1, 1) + y + torch.roll(y, -1, 1)) / 3
    m = (torch.rand(B, 1, generator=g, device=x.device) < 0.3).float()
    y = m * sm + (1 - m) * y
    y = y + 0.01 * torch.randn_like(y)
    return y.clamp(0, 1)


def train_jepa(X, h=32, k=3, epochs=300, lr=1e-3, bs=512, tau=0.99,
               wvar=25.0, wcov=1.0, seed=0):
    torch.manual_seed(seed)
    Xt = torch.tensor(X, device=DEV)
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    enc, pred = Enc(X.shape[1], h, k).to(DEV), Pred(h, k).to(DEV)
    tgt = Enc(X.shape[1], h, k).to(DEV)
    tgt.load_state_dict(enc.state_dict())
    for p in tgt.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=lr)
    N = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(N, device=DEV, generator=g)
        for s in range(0, N, bs):
            b = perm[s:s + bs]
            if len(b) < 32:
                continue
            v1, v2 = augment(Xt[b], g), augment(Xt[b], g)
            z1 = enc(v1)
            with torch.no_grad():
                z2 = tgt(v2)
            p1 = pred(z1)
            sim = 2 - 2 * F.cosine_similarity(p1, z2, dim=1).mean()
            var = torch.mean(F.relu(1 - torch.sqrt(z1.var(0) + 1e-4)))
            zc = z1 - z1.mean(0)
            cov = zc.T @ zc / max(len(b) - 1, 1)
            off = cov - torch.diag(torch.diag(cov))
            covl = (off ** 2).sum() / z1.shape[1]
            loss = sim + wvar * var + wcov * covl
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for pt, po in zip(tgt.parameters(), enc.parameters()):
                    pt.mul_(tau).add_(po.detach(), alpha=1 - tau)
    with torch.no_grad():
        Z = enc(Xt).cpu().numpy()
    return enc, Z


# ------------------------------------------------------------------ 主流程
def main():
    t0 = time.time()
    X, Y = load_pwdb()
    targets = list(Y.keys())
    print(f"PWDB n={len(X)} dim={X.shape[1]} device={DEV}")

    def score(Fm):
        return {c: probe(np.asarray(Fm, float), Y[c].astype(float)) for c in targets}

    def multi_seed(fn, y_score=True):
        """3 个种子：返回中位数指标 + 中位 PR + 参数量。"""
        res = []
        for s in SEEDS:
            net, r2, Z = fn(s)
            res.append(dict(r2=r2, Z=Z, params=nparam(net)))
        r2 = float(np.median([r["r2"] for r in res]))
        pr = float(np.median([effrank(r["Z"]) for r in res]))
        Sc = [score(r["Z"]) for r in res] if y_score else None
        if y_score:
            sc = {c: float(np.median([s[c] for s in Sc])) for c in targets}
            rho_m = float(np.mean(list(sc.values())))
        else:
            sc, rho_m = {}, np.nan
        return dict(recon_r2=r2, effrank=pr, params=res[0]["params"],
                    **{f"rho_{c}": sc.get(c, np.nan) for c in targets},
                    rho_mean=rho_m)

    rows = []
    # ---------- 参考点 ----------
    s = score(X)
    rows.append(dict(model="raw256", k=256, h="-", params=256 * len(targets),
                     lam=0.0, recon_r2=1.0, effrank=effrank(X),
                     **{f"rho_{c}": s[c] for c in targets},
                     rho_mean=float(np.mean(list(s.values())))))
    print(f"  raw256            rho_mean={np.mean(list(s.values())):.3f}")

    for k in [1, 2, 3, 4, 6, 8, 16, 32]:
        p = PCA(k, random_state=0).fit(X)
        P = p.transform(X)
        s = score(P)
        r2 = 1 - ((X - p.inverse_transform(P)) ** 2).sum() / ((X - X.mean(0)) ** 2).sum()
        rows.append(dict(model="pca", k=k, h="-", params=int(256 * k), lam=0.0,
                         recon_r2=float(r2), effrank=effrank(P),
                         **{f"rho_{c}": s[c] for c in targets},
                         rho_mean=float(np.mean(list(s.values())))))
        print(f"  pca k={k:<3d}        R2={r2:.3f} rho_mean={np.mean(list(s.values())):.3f}")

    # ---------- 瓶颈 AE：k 扫（h=32）----------
    for k in [1, 2, 3, 4, 6, 8, 16]:
        m = multi_seed(lambda s, k=k: train_ae(X, 32, k, seed=s))
        rows.append(dict(model="ae_h32", k=k, h=32, **m))
        print(f"  ae h=32 k={k:<3d}  params={m['params']:>6d} R2={m['recon_r2']:.3f} "
              f"PR={m['effrank']:.2f} rho_mean={m['rho_mean']:.3f}")

    # ---------- 瓶颈 AE：h 扫（k=3 固定）----------
    for h in [4, 8, 16]:
        m = multi_seed(lambda s, h=h: train_ae(X, h, 3, seed=s))
        rows.append(dict(model=f"ae_h{h}", k=3, h=h, **m))
        print(f"  ae h={h:<3d} k=3    params={m['params']:>6d} R2={m['recon_r2']:.3f} "
              f"rho_mean={m['rho_mean']:.3f}")

    # ---------- 稀疏 SAE：仅在 k 足以承载稀疏性时 ----------
    for h, k in [(128, 8), (128, 32), (16, 8)]:
        m = multi_seed(lambda s, h=h, k=k: train_ae(X, h, k, lam=1e-3,
                                                    epochs=1200, seed=s, sparse=True))
        rows.append(dict(model=f"sae_h{h}", k=k, h=h, lam=1e-3, **m))
        print(f"  sae h={h:<3d} k={k:<3d} R2={m['recon_r2']:.3f} "
              f"rho_mean={m['rho_mean']:.3f}")

    # ---------- 失效模式：小瓶颈 + ReLU-L1 稀疏 ----------
    for k in [2, 3, 4]:
        res = []
        for s in SEEDS:
            net, r2, Z = train_ae(X, 16, k, lam=1e-3, epochs=1200, seed=s, sparse=True)
            res.append((r2, int((Z.std(0) > 1e-6).sum())))
        rows.append(dict(model="sae_collapse", k=k, h=16, lam=1e-3,
                         alive=int(np.median([a for _, a in res])),
                         recon_r2=float(np.median([r for r, _ in res])),
                         effrank=np.nan, rho_mean=np.nan))
        print(f"  [collapse] k={k} R2={np.median([r for r,_ in res]):+.3f} "
              f"alive={rows[-1]['alive']}/{k}")

    # ---------- JEPA ----------
    for h, k in [(16, 2), (16, 3), (16, 4), (32, 3), (32, 8), (32, 16)]:
        Zs, Ps, prm = [], [], 0
        for s in SEEDS:
            enc, Z = train_jepa(X, h, k, seed=s)
            Zs.append(Z)
            Ps.append(score(Z))
            prm = nparam(enc) + sum(p.numel() for p in Pred(h, k).parameters())
        sc = {c: float(np.median([p[c] for p in Ps])) for c in targets}
        rows.append(dict(model="jepa", k=k, h=h, params=prm, lam=0.0,
                         recon_r2=np.nan,
                         effrank=float(np.median([effrank(z) for z in Zs])),
                         **{f"rho_{c}": sc[c] for c in targets},
                         rho_mean=float(np.mean(list(sc.values())))))
        print(f"  jepa h={h:<3d} k={k:<3d} params={prm:>6d} "
              f"PR={rows[-1]['effrank']:.2f} rho_mean={rows[-1]['rho_mean']:.3f}")

    # ---------- 输入分辨率：真正决定参数量的轴 ----------
    res_rows = []
    for R in [32, 64, 128, 256]:
        Xr = X if R == 256 else resample(X, R)
        for h in [8, 16]:
            m = multi_seed(lambda s, Xr=Xr, h=h: train_ae(Xr, h, 3, seed=s))
            res_rows.append(dict(R=R, h=h, k=3, params=m["params"],
                                 recon_r2=m["recon_r2"], effrank=m["effrank"],
                                 **{f"rho_{c}": m[f"rho_{c}"] for c in targets},
                                 rho_mean=m["rho_mean"]))
            print(f"  res R={R:<4d} h={h:<3d} params={m['params']:>6d} "
                  f"R2={m['recon_r2']:.3f} rho_mean={m['rho_mean']:.3f}")

    key = ["model", "k", "h", "params", "lam", "alive", "recon_r2", "effrank",
           "rho_mean"] + [f"rho_{c}" for c in targets]
    with open(os.path.join(DATA, "tiny_capacity.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=key, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(DATA, "tiny_resolution.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["R", "h", "k", "params", "recon_r2",
                                           "effrank", "rho_mean"] +
                           [f"rho_{c}" for c in targets], extrasaction="ignore")
        w.writeheader()
        w.writerows(res_rows)

    # 保存 k=3 的 AE 供干预实验复用
    net, r2, Z = train_ae(X, 32, 3, seed=0)
    with torch.no_grad():
        rec, _ = net(torch.tensor(X, device=DEV))
    np.savez(os.path.join(DATA, "tiny_ae_k3.npz"), Z=Z, X=X,
             rec=rec.cpu().numpy(), r2=r2,
             W_enc=net.enc[2].weight.detach().cpu().numpy(),
             W_dec=net.dec[0].weight.detach().cpu().numpy(),
             keys=np.array(targets, dtype=object),
             Y=np.array([Y[c] for c in targets]))
    print(f"\n写入 tiny_capacity.csv ({len(rows)} 行) / tiny_resolution.csv "
          f"({len(res_rows)} 行) / tiny_ae_k3.npz  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
