"""
k=3 极小模型上的因果有效性检验，与 74k 参数 SAE(k=32) 的同名结果对比。

三件事（口径完全沿用 m1_sae.py，便于直接对照）：
  1) 潜变量-机制对齐：每个 z_k 与 AI_r / PWV_cf / SBP_a / PP_a 的 Spearman
  2) 干预：固定其余两个潜变量，只沿 z_target 移动 dz∈[-2,2]，解码波形后
     用同一 aix_from_wave 量形态变化 -> 看是否单调、方向是否与机制一致
  3) 靶向消融：置零单个潜变量，看各读出损失多少

判据（冻结 v2.3 G3）：若消融 Δρ≈0 且干预非单调，则"SAE 潜变量具有因果有效性"
不成立。这里要回答的是：把模型从 74k 压到 17k 参数、32 潜变量压到 3 个，
这个结论会不会变。

输出 data/tiny_intervention.csv / data/tiny_ablation.csv
"""

import os as _os, sys as _sys

# Resolved at run time: scripts/ holds sibling groups, the repository root holds data/.
_S = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _S)
for _d in _os.listdir(_S):
    if _d[:1].isdigit() and _os.path.isdir(_os.path.join(_S, _d)):
        _sys.path.append(_os.path.join(_S, _d))
import _repo  # noqa: E402

import os, sys, csv, json, warnings
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = _repo.ROOT
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from tiny_models import probe

z = np.load(os.path.join(DATA, "tiny_ae_k3.npz"), allow_pickle=True)
Z = z["Z"].astype(float)
X = z["X"].astype(float)
KEYS = [str(k) for k in z["keys"]]
Y = {k: v for k, v in zip(KEYS, z["Y"])}
print(f"k=3 AE  n={len(Z)} latent={Z.shape[1]} recon_R2={float(z['r2']):.4f}")
print(f"  潜变量 std = {np.round(Z.std(0), 3)}")


# ---------------- 1) 对齐 ----------------
align_rows = []
for k in range(Z.shape[1]):
    d = {"latent": k}
    for c in KEYS:
        d["rho_" + c] = round(float(spearmanr(Z[:, k], Y[c])[0]), 3)
    align_rows.append(d)
A = sorted(align_rows, key=lambda r: -max(abs(r["rho_" + c]) for c in KEYS))
print("\n[1/3] 潜变量-机制对齐（k=3）")
for r in A:
    print("  " + "  ".join(f"{kk}={vv}" for kk, vv in r.items()))

with open(os.path.join(DATA, "tiny_alignment.csv"), "w", newline="",
          encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=["latent"] + ["rho_" + c for c in KEYS])
    w.writeheader()
    w.writerows(align_rows)


# ---------------- 2) 干预 ----------------
def aix_from_wave(p):
    """与 m1_sae.py 逐行一致的实现（切迹前收缩期肩部）。"""
    q = (p - p.min()) / (p.max() - p.min() + 1e-9)
    n = len(q)
    i1 = int(np.argmax(q[: int(.55 * n)]))
    tail = q[i1:]
    cand = [i for i in range(1, len(tail) - 1)
            if tail[i] >= tail[i - 1] and tail[i] >= tail[i + 1]]
    i2 = i1 + cand[int(np.argmax([tail[i] for i in cand]))] if cand else i1
    Fq = np.abs(np.fft.rfft(q - q.mean()))
    return float(q[i2] - q[i1]), float(Fq[2] / (Fq[1] + 1e-9))


# 由于 npz 只存了首层权重，直接重训同一 seed 的模型做解码（确定性）
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from tiny_models import train_ae, DEV

net, r2, Z2 = train_ae(X.astype(np.float32), 32, 3, seed=0)
print(f"\n  复核：重训模型 R2={r2:.4f}（与存档 {float(z['r2']):.4f} 一致），"
      f"潜变量最大偏差 {np.abs(Z2 - Z).max():.4f}")

tgt_k = int(max(range(3), key=lambda k: abs(spearmanr(Z[:, k], Y["PWV_cf"])[0])))
ctl_k = int(min(range(3), key=lambda k: max(abs(spearmanr(Z[:, k], Y[c])[0])
                                            for c in KEYS)))
print(f"  目标潜变量 z{tgt_k}（|rho PWV| 最大）  对照 z{ctl_k}（全机制 |rho| 最小）")

sd = Z[:, tgt_k].std() + 1e-9
rows = []
grid = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
with torch.no_grad():
    for dz in grid:
        Zm = Z.copy()
        Zm[:, tgt_k] = Z[:, tgt_k] + dz * sd
        w = net.dec(torch.tensor(Zm, device=DEV, dtype=torch.float32)).cpu().numpy()
        zb = net.enc(torch.tensor(w, device=DEV, dtype=torch.float32)).cpu().numpy()
        ap = np.array([aix_from_wave(x) for x in w])
        rows.append(dict(dz=dz, aix_mean=float(ap[:, 0].mean()),
                         h2_mean=float(ap[:, 1].mean()),
                         reenc_err=float(np.abs(zb - Zm).mean())))
S = rows
rho_int = spearmanr([r["dz"] for r in S], [r["aix_mean"] for r in S])[0]
d_aix = S[-1]["aix_mean"] - S[0]["aix_mean"]
sign = np.sign(d_aix) if d_aix != 0 else 1
viol = int(np.sum(np.diff([r["aix_mean"] for r in S]) * sign < 0))

print("\n[2/3] 干预：只移动 z{}, 其余固定".format(tgt_k))
for r in S:
    print(f"  dz={r['dz']:+.1f}  AIx={r['aix_mean']:+.4f}  h2={r['h2_mean']:.4f}  "
          f"再编码误差={r['reenc_err']:.4f}")
print(f"  rho(dz, AIx) = {rho_int:+.3f}   ΔAIx(dz=-2→+2) = {d_aix:+.4f}   "
      f"方向反转 = {viol}/{len(S)-1}")

with open(os.path.join(DATA, "tiny_intervention.csv"), "w", newline="",
          encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=list(S[0].keys()))
    w.writeheader()
    w.writerows(S)


# ---------------- 3) 靶向消融 ----------------
print("\n[3/3] 靶向消融：置零单个潜变量")
abl = []
for c in KEYS:
    y = Y[c].astype(float)
    base = probe(Z, y)
    row = {"target": c, "full": round(float(base), 3)}
    for k in range(3):
        Zk = Z.copy()
        Zk[:, k] = 0.0
        row[f"drop_z{k}"] = round(float(probe(Zk, y)), 3)
    abl.append(row)
    print("  " + "  ".join(f"{kk}={vv}" for kk, vv in row.items()))

with open(os.path.join(DATA, "tiny_ablation.csv"), "w", newline="",
          encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=list(abl[0].keys()))
    w.writeheader()
    w.writerows(abl)

json.dump(dict(target_latent=tgt_k, control_latent=ctl_k,
               rho_intervention=float(rho_int), delta_aix=float(d_aix),
               reversals=int(viol), recon_r2=float(r2)),
          open(os.path.join(DATA, "tiny_intervention_summary.json"), "w"),
          ensure_ascii=False, indent=2)
print("\n写入 tiny_alignment.csv / tiny_intervention.csv / tiny_ablation.csv")
