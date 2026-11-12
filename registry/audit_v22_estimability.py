"""v2.2 构成的可估性审查 —— 检验「份额能不能当成一个数报出来」。

要回答的问题
------------
v2.2 的阳性对照（audit_v22_power.py）留下两个红灯：
  (1) SNR=0 的纯噪声特征给出 desc z=+2.15 / device z=-1.68 / mech z=-0.34，
      三族偏离 0 的幅度互不一致，说明**标签置换零分布没校准**；
  (2) 植入 stiffness 单簇方向、SNR=1 时目标簇 z 只有 +0.17，**定位能力缺失**。
两者都指向同一个怀疑：构成 comp_c = Σd_f φ_fc / Σd_f R²_f 是两个小量之商，
当 d 集中在少数特征上（有效特征数 n_eff 很小）时，它的抽样波动可能大到让
「descriptor 比 mechanism 高」这种排序根本不可判。

怎么测
------
一条自助法轴，另加两项数值检查（都不依赖那个可疑的标签置换零分布）：

  特征轴  w_b ~ Multinomial(F; d/Σd)       重抽"哪几个 SAE 特征承载权重"

  注意力：抽样概率必须取 d 的经验分布，不能取均匀分布。均匀重抽等于把权重摊到全部
  128 个特征上，得到的分布中心会离开点估计（PWDB 实测 Δ 从 −0.057 漂到 +0.015），
  那不是置信区间，是另一个模型的取值。

  样本轴被弃用：重抽行会改变 C 的零空间方向，而 L1 的相关矩阵条件数 1e13、
  被权重压住的几个量 VIF 达 1e14（audit_v22_ridge.py 实测），沿这些方向
  g^T inv(C) g 会被 ridge 放大约 1e9 倍，coverage 在 13%–97% 之间乱跳。
  那不是抽样噪声，是数值病态，因此样本轴不作为区间来源。

  数值检查一  把 ridge 从 pinv 扫到 1e-1（audit_v22_ridge.py），看分解是否良定义
  数值检查二  coverage 的零线：把 A 的行随机打乱，切断表征与 L1 的对应

关键判据是**差值的区间**，不是各自的区间：
  Δ = comp_desc − comp_mech 的 95% 区间是否排除 0
只有排除 0，才可以说"这一份权重更偏重述而不是机理"。
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

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phys_layer as PL          # noqa: E402
import probe_mediator as PM      # noqa: E402

DATA = PM.DATA
FAMS = ("mechanism", "descriptor", "device")
B = 300


def fam_of_comp(comp, cs):
    out = np.zeros(3)
    for i, c in enumerate(cs):
        out[FAMS.index(PL.family_of(c))] += comp[i]
    return out


def split(sd, phi, r2, w):
    cov = float((w * r2).sum())
    comp = (phi * w).sum(1) / max(cov, 1e-12)
    return cov, comp


def shap_with(sd, inv, G):
    """用给定的 C 逆矩阵块重算 φ 与 R²（供样本轴自助复用）。"""
    r = {}
    F = G.shape[1]
    for S in sd.subs:
        rows = sd.rows[S]
        if len(rows) == 0:
            r[S] = np.zeros(F)
            continue
        g = G[rows]
        r[S] = np.einsum("if,if->f", g, inv[S] @ g)
    phi = np.zeros((len(sd.cs), F))
    for c in range(len(sd.cs)):
        for S in sd.subs:
            if c in S:
                continue
            phi[c] += sd.wgt[(c, S)] * (r[tuple(sorted(S + (c,)))] - r[S])
    return np.clip(phi, 0, None), r[tuple(range(len(sd.cs)))]


def main():
    z = np.load(os.path.join(DATA, "probe_mediator_v22.npz"), allow_pickle=True)
    qn = list(z["qnames"])
    cfgs = sorted({k.rsplit("_", 2)[0] for k in z.files if k.endswith("_A")})
    cfgs = [c for c in cfgs if c not in ("clusters", "qnames")]

    rows = []
    print(f"{'配置':<18s}{'种':>3s}{'覆盖%':>7s}{'机理':>7s}{'重述':>7s}{'噪声':>7s}"
          f"{'n_eff':>7s}{'Δ(re−me) 95%CI':>20s}{'可判':>5s}{'零线%':>7s}")
    for cfg in cfgs:
        for s in (0, 1, 2):
            key = f"{cfg}_s{s}"
            if key + "_A" not in z.files:
                continue
            A = z[key + "_A"].astype(np.float64)
            V = z[key + "_V"].astype(np.float64)
            c = z[key + "_c"].astype(np.float64)
            V = np.where(np.isfinite(V), V, np.nanmedian(V, 0))
            n, F = A.shape
            d = np.abs(c * A.mean(0))
            w0 = d / d.sum()
            neff = 1.0 / (w0 ** 2).sum()

            sd = PM.ShapleyAttrib(V, qn)
            Az = (A - A.mean(0)) / (A.std(0) + 1e-12)
            G = sd.Z.T @ Az / n
            phi, r2 = shap_with(sd, sd.inv, G)
            cov, comp = split(sd, phi, r2, w0)
            obs = fam_of_comp(comp, sd.cs)

            rng = np.random.default_rng(4242)
            # 特征轴
            df = np.zeros((B, 3))
            for b in range(B):
                i = rng.choice(F, F, p=w0)
                w = np.bincount(i, minlength=F).astype(float)
                w /= w.sum()
                _, cp = split(sd, phi, r2, w)
                df[b] = fam_of_comp(cp, sd.cs)
            delta = df[:, 1] - df[:, 0]
            lo, hi = np.percentile(delta, [2.5, 97.5])
            ok = (lo > 0) or (hi < 0)

            # coverage 零线：打断 A 与 V 的对应
            Gp = sd.Z.T @ Az[rng.permutation(n)] / n
            php, r2p = shap_with(sd, sd.inv, Gp)
            base = float((w0 * r2p).sum())

            print(f"{cfg:<18s}{s:>3d}{100*cov:>7.1f}{obs[0]:>7.3f}{obs[1]:>7.3f}"
                  f"{obs[2]:>7.3f}{neff:>7.2f}{'[%+.2f, %+.2f]' % (lo, hi):>20s}"
                  f"{'是' if ok else '否':>5s}{100*base:>7.2f}")
            rows.append(dict(cfg=cfg, seed=s, coverage=round(cov, 5),
                             mech=round(obs[0], 4), desc=round(obs[1], 4),
                             device=round(obs[2], 4), n_eff=round(neff, 3),
                             d_lo=round(float(lo), 4), d_hi=round(float(hi), 4),
                             estimable=int(ok), cov_null=round(base, 5)))
        print()

    fn = os.path.join(DATA, "audit_v22_estimability.csv")
    with open(fn, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("=" * 100)
    print(f"Δ=comp_desc−comp_mech 的 95% 区间排除 0 的配置："
          f"{sum(r['estimable'] for r in rows)}/{len(rows)}")
    print(f"平均有效特征数 n_eff = {np.mean([r['n_eff'] for r in rows]):.2f} / 128")
    print(f"coverage 中位数 {100*np.median([r['coverage'] for r in rows]):.1f}%  "
          f"vs 打断对应后的零线中位数 {100*np.median([r['cov_null'] for r in rows]):.1f}%")
    print(f"-> {fn}")


if __name__ == "__main__":
    main()
