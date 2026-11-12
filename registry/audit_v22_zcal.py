"""v2.2 标签置换 z 的校准审查 —— 直接测误报率，逐源测偏移。

为什么必须单独测
----------------
audit_v22_power.py 里出现过两个互相矛盾的数：同为纯噪声特征，一次给出 device z=+2.86，
另一次给出 −1.68。同一个零假设下两次抽样差 4.5σ，只有一个解释：z 的方差被低估，
或者它的中心根本不在 0。标签置换只覆盖「簇标签的随机性」，不覆盖「特征权重 d 的抽样
噪声」，而 d 往往只压在少数特征上（可估性审查实测 n_eff 中位数 13/128）。

这件事不能靠一次抽样判断。这里把零分布本身当成被测量对象：固定 d 的形状，
独立重抽表征 A，重复 M 次，看 z 的**分布**是不是 N(0,1)。

为什么偏移不是"抽样噪声"而是结构性的
------------------------------------
真实分簇里，同一簇的物理量彼此高度相关（17 号实测 reflection~dicrotic r=+1.00）。
用 5 个高度相关的列去回归，拿到的 R² 低于「5 个随机列」，而标签置换正好保持每簇的
**大小**、打散它的**内部相关**。于是置换零分布系统性地高估每个簇的解释力，且高估
幅度与该簇的内部冗余度成正比 —— 冗余度最高的机理簇被压得最低（z 偏负），
冗余度最低的设备簇反而偏高。这不是"数据有偏好"，是"分簇比随机分组更相关"这件
永远成立的事被算进了统计量。

技术上要快
----------
attr_null 每次调用都重建 200 组置换标签的协方差逆矩阵，那一步占了几乎全部时间。
但置换标签与 G、d 无关 —— 同一份 V、同一个种子，逆矩阵是常数。于是预先建一次
NullBank，之后每次评估只剩 64 个子集的小矩阵乘法，单次 z 从 ~15s 降到 ~2ms。
这样才能跑 M=200 次而不是 M=2 次。

四个实验
--------
E1 逐源偏移  每源各自建银行，A 与 V 独立，M=200，给出三族的 μ/σ/|z|>2 比例
E2 集中度    d 的有效特征数取 8 / 44 / 128，看校准是否随 d 集中度恶化
E3 功效      植入 noise 簇方向，M=100，看目标族 z 与检出率（覆盖度 70–90% 时）
E4 零模型    d 取真实配置的 |c·ā| 形状但 A 与 V 独立，看"没学到的模型"能刷出多少 z
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
NPERM = 200
SRCS = ("PWDB", "CM", "VDB", "MIMIC")


class NullBank:
    """预先把 NPERM 组置换标签的 C 逆矩阵块建好，供反复评估。"""

    def __init__(self, sd, nperm=NPERM, seed=0):
        self.cs = sd.cs
        self.subs = sd.subs
        self.wgt = sd.wgt
        self.rows = sd.rows
        self.names = sd.names
        self.fams = [PL.family_of(c) for c in sd.cs]
        sizes = [len([1 for x in sd.labels if x == c]) for c in sd.cs]
        rng = np.random.default_rng(20261103 + seed)
        self.banks = []
        for _ in range(nperm):
            perm = rng.permutation(len(self.names))
            lab = np.empty(len(self.names), object)
            done = 0
            for c, s in zip(self.cs, sizes):
                lab[perm[done:done + s]] = c
                done += s
            rows_b, inv_b = {}, {}
            for S in self.subs:
                cols = [j for c in S for j, nm in enumerate(lab) if nm == self.cs[c]]
                rows_b[S] = np.array(cols, int)
                if cols:
                    A = sd.C[np.ix_(cols, cols)]
                    inv_b[S] = np.linalg.pinv(A + 1e-9 * np.eye(len(cols)))
            self.banks.append((rows_b, inv_b))

    @staticmethod
    def _phi(rows_b, inv_b, subs, wgt, cs, G):
        r = {}
        F = G.shape[1]
        for S in subs:
            rr = rows_b[S]
            if len(rr) == 0:
                r[S] = np.zeros(F)
                continue
            g = G[rr]
            r[S] = np.einsum("if,if->f", g, inv_b[S] @ g)
        phi = np.zeros((len(cs), F))
        for c in range(len(cs)):
            for S in subs:
                if c in S:
                    continue
                phi[c] += wgt[(c, S)] * (r[tuple(sorted(S + (c,)))] - r[S])
        return np.clip(phi, 0, None), r[tuple(range(len(cs)))]

    def _fam(self, comp):
        out = np.zeros(3)
        for i, f in enumerate(FAMS):
            out[i] = comp[[j for j, g in enumerate(self.fams) if g == f]].sum()
        return out

    def eval(self, G, d, sd):
        w = d / max(d.sum(), 1e-12)
        phi, r2 = self._phi(self.rows, sd.inv, self.subs, self.wgt, self.cs, G)
        cov = float((w * r2).sum())
        obs = self._fam((phi * w).sum(1) / max(cov, 1e-12))
        null = np.zeros((len(self.banks), 3))
        for b, (rows_b, inv_b) in enumerate(self.banks):
            ph, r2b = self._phi(rows_b, inv_b, self.subs, self.wgt, self.cs, G)
            cv = max(float((w * r2b).sum()), 1e-12)
            null[b] = self._fam((ph * w).sum(1) / cv)
        z = (obs - null.mean(0)) / (null.std(0) + 1e-12)
        return cov, obs, null, z


def make_d(F, neff, seed=0, place=None):
    """有效特征数约为 neff 的权重（几何衰减）；place 指定压在哪些列上。"""
    rng = np.random.default_rng(seed)
    k = max(2, int(round(neff)))
    v = np.zeros(F)
    v[:k] = 0.62 ** np.arange(k)
    if place is not None:
        out = np.zeros(F)
        out[place] = v[:len(place)]
        return out / out.sum()
    v = v[rng.permutation(F)]
    return v / v.sum()


def stats(name, zs):
    zs = np.asarray(zs)
    txt = []
    for i, f in enumerate(FAMS):
        c = zs[:, i]
        txt.append(f"{f[:4]} μ={c.mean():+.2f} σ={c.std():.2f} FPR={100*np.mean(np.abs(c) > 2):4.1f}%")
    print(f"    {name:<10s}" + " | ".join(txt))
    return zs.mean(0), zs.std(0), [float(np.mean(np.abs(zs[:, i]) > 2)) for i in range(3)]


def main():
    rows = []
    offsets = {}
    for src in SRCS:
        d0 = PM.load_source(src)
        V = d0["V"].astype(np.float64)
        V = np.where(np.isfinite(V), V, np.nanmedian(V, 0))
        qn = list(d0["qnames"])
        sd = PM.ShapleyAttrib(V, qn)
        sd.C = sd.Z.T @ sd.Z / len(sd.Z)
        n, p = sd.Z.shape
        F = 128
        cols = [j for j, q in enumerate(qn) if q in PL.TABLE["noise"]["feats"]]
        u = np.zeros(p); u[cols] = 1.0
        u /= np.linalg.norm(u)
        sig = sd.Z @ u
        bank = NullBank(sd)
        print(f"[{src}] n={n} p={p} 簇={len(PL.TABLE)} 置换银行={len(bank.banks)}×{len(bank.subs)}")

        print("  E1 零校准（A⊥V，M=200，名义 FPR=5%）")
        rng = np.random.default_rng(7)
        zs = []
        for _ in range(200):
            A = rng.normal(size=(n, F))
            Az = (A - A.mean(0)) / (A.std(0) + 1e-12)
            zs.append(bank.eval(sd.Z.T @ Az / n, make_d(F, 8, seed=1), sd)[3])
        mu, sdd, fpr = stats("n_eff≈8", zs)
        offsets[src] = mu
        zs44 = []
        rng = np.random.default_rng(7)
        for _ in range(100):
            A = rng.normal(size=(n, F))
            Az = (A - A.mean(0)) / (A.std(0) + 1e-12)
            zs44.append(bank.eval(sd.Z.T @ Az / n, make_d(F, 44, seed=1), sd)[3])
        stats("n_eff≈44", zs44)

        print("  E3 功效（植入 noise 簇方向，d 压在前 5 列，M=100）")
        dz = np.zeros(F)
        dz[:5] = [300, 200, 150, 100, 50]
        for snr in (1.0, 2.0):
            rng = np.random.default_rng(11)
            zs2, cv2 = [], []
            for _ in range(100):
                A = rng.normal(size=(n, F))
                for j in range(5):
                    A[:, j] = snr * sig + rng.normal(size=n)
                Az = (A - A.mean(0)) / (A.std(0) + 1e-12)
                cv, obs, nl, z = bank.eval(sd.Z.T @ Az / n, dz, sd)
                zs2.append(z); cv2.append(cv)
            print(f"    SNR={snr} 覆盖={100*np.mean(cv2):.0f}%")
            stats("原 z", zs2)
            stats("减偏移", np.asarray(zs2) - mu)
        print()
        rows.append(dict(source=src, n=n, mu_mech=round(mu[0], 3), mu_desc=round(mu[1], 3),
                         mu_device=round(mu[2], 3), sd_mech=round(sdd[0], 3),
                         sd_desc=round(sdd[1], 3), sd_device=round(sdd[2], 3),
                         fpr_mech=round(fpr[0], 3), fpr_desc=round(fpr[1], 3),
                         fpr_device=round(fpr[2], 3)))

    fn = os.path.join(DATA, "audit_v22_zcal.csv")
    with open(fn, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    print("=" * 96)
    print("逐源零分布（名义：μ=0 σ=1 FPR=5%）")
    print(f"{'源':<8s}{'n':>6s}{'μ_机':>7s}{'μ_重':>7s}{'μ_噪':>7s}{'σ_机':>6s}{'σ_重':>6s}"
          f"{'σ_噪':>6s}{'FPR_机':>8s}{'FPR_重':>8s}{'FPR_噪':>8s}")
    for r in rows:
        print(f"{r['source']:<8s}{r['n']:>6d}{r['mu_mech']:>+7.2f}{r['mu_desc']:>+7.2f}"
              f"{r['mu_device']:>+7.2f}{r['sd_mech']:>6.2f}{r['sd_desc']:>6.2f}"
              f"{r['sd_device']:>6.2f}{r['fpr_mech']:>8.1%}{r['fpr_desc']:>8.1%}{r['fpr_device']:>8.1%}")
    print(f"-> {fn}")


if __name__ == "__main__":
    main()
