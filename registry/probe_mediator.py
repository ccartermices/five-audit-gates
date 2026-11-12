"""用物理量中介层解剖四源模型：它在想什么、想得对不对。

与 probe_what.py 的分工
-----------------------
probe_what 只在 PWDB 上做，归因靶子是**仿真真值参数**（104 列）。那只回答"仿真里对不对"，
因为真实队列永远拿不到生成参数。
本文件把归因靶子换成**物理量中介层 L1**（phys_layer.py 的 30 个无量纲可测量（v2.3）），
四源同一实现、同一口径，于是四个库的模型第一次可以用同一把尺子比。

要回答的三个问题
----------------
1. 训练成不成功：单源训练 + 受试者分折的 OOF rho。跨库不混训（11 号 §4 的结论）。
2. 它在想什么：SAE 特征归因到 L1 的**正交主成分**，再把每一维的方差按载荷回投到
   6 个机制簇（见 ortho_dims / cluster_loading）。v2.2 起份额必须带置换零分布的 z，
   不带 z 的份额不构成结论。
3. 想得对不对：**MSC 符号一致性**（见下）+ 与 L1-Ridge 上限、噪声单簇基线的对照。

MSC：本文件新增的判据
--------------------
相关不等于在用（11 号 §3.4 已证），在用也不等于用对方向。设
  c_f   = 特征 f 的单位激活对预测的贡献（输出头对表征线性，可解析求出）
  sigma = sign(rho(a_f, m*))，m* 是 f 归因最强的物理量
  beta  = sign(rho(m*, y))，该物理量对血压的真实边际方向
则模型通过 f 隐含的「m* -> 血压」方向为 sign(c_f)*sigma，必须与 beta 同号才算通过。
不通过意味着：这个特征确实在驱动预测，但它对物理量做的是**反向**推理（混淆或非单调），
这种特征不能当作"依据正确"的证据。通过率按簇统计，是本报告判定 A/B/C 的核心。

判据（v2.2 起按 z，事前声明，写入 12 / 18 号文档）
--------------------------------------------------
A 读机制     OOF rho >= 0.35 且 机理 z > +2 且 设备 z < -2
B 拟合噪声   设备 z > +2
C 没学到     OOF rho < 0.35
D 未检出     rho 够，但机理 z 不显著

为什么不用份额的绝对值当判据
----------------------------
v2.1 及以前用 "机理份额 >= 0.60" 判 A。17 号证明这个门槛没有推断内容：机理族在
8 个簇里占 5 个，随机分配下的期望份额就是 5/8 = 62.5%，实测 52.7–64.1% 正落在
基线附近，四源 z 全部 <= +0.38。份额高低本身也不带抽样误差，无法与"偶然"区分。
v2.2 因此改问：这个份额超出行置换零分布几个标准差。低于 2 一律读作"未检出"。

噪声簇是唯一的"非生理"家族：它由 5 阶以上谐波构成，在窄带脉搏波里基本是量化/采样率
痕迹。单源训练时设备是常数、噪声不与血压相关，因此它若显著占高份额只能是过拟合。
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
import csv
import itertools
import math
import os
import sys

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phys_layer as PL           # noqa: E402
from probe_what import (ARCHS, DEV, assoc, fit_model,  # noqa: E402
                        train_sae)

ROOT = _repo.ROOT
DATA = os.path.join(ROOT, "data")

SRC = ["PWDB", "VDB", "MIMIC", "CM"]
THR_RHO = 0.35      # C 的下界
THR_PHYS = 0.60     # 旧口径（v2.1 及以前）：机理份额下界，已被 THR_Z 取代
THR_DEV = 0.50      # 旧口径：设备噪声份额下界，已被 THR_Z 取代
THR_Z = 2.6         # v2.2：构成必须超过标签置换零分布 2.6 个标准差才算"检出"
THR_COV = 0.30      # v2.2：L1 至少解释 30% 的因果权重，否则判定无意义

# 闸门判据的冻结时点与阈值常量（27 号：投稿层级评估要求 §4.7 的数字可溯源）
# 这些常量只在 27 号的 §4.7 表格里出现，正文必须走 registry 引用，不得手写。
THR_ICC = 0.80         # G3 采样单元闸门：量在报告单元上的 ICC 门槛
THR_JACCARD = 0.50     # G5 归因稳定闸门：跨种子 top-12 的 Jaccard 门槛
THR_FPR_FACTOR = 2     # G4 零分布闸门：实际误报率不得超名义率的两倍
THR_NULL_SD = 0.5      # G4 零分布闸门：零分布均值偏移不得超 0.5 SD
THR_PROP_PCT = 50      # G5 簇级判据：过半配置达标
GATE_FIXED_YEAR_G1 = 2026   # G1 判据在正文撰写前冻结
GATE_FIXED_YEAR_G4 = 2026   # G4 判据为事后设定（用于在正文中如实标注）
GATE_FIXED_YEAR_G5 = 2026   # G5 判据在种子稳定性实验前冻结


# ---------------------------------------------------------------- 数据
def _rd(name):
    with open(os.path.join(DATA, name), encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _col(rows, key):
    out = []
    for r in rows:
        v = r.get(key, "")
        try:
            f = float(v)
        except (TypeError, ValueError):
            f = np.nan
        out.append(f if np.isfinite(f) else np.nan)
    return np.asarray(out, np.float64)


def load_source(name, keep_nan=False):
    """-> dict(X, y, groups, V, qnames, dropped, meta)

    X 逐周期 min-max：与 06/07 的 UNCAL 协议一致，也是四设备唯一可比口径。
    groups 用于受试者分折：MIMIC 425 人 12750 段、CM 57 人 334 段，若按行分折，
    同一个人会同时出现在训练与测试里，rho 会被抬高。

    meta 是**与 X/y/V 逐行对齐**的分层变量（目前只有 VDB 有）。分层重跑最容易出
    的错是切分掩码没跟着质量门走，导致标签与波形错位——所以对齐必须在这里做，
    由构造保证，而不是让下游脚本各自重算。
    """
    meta = {}
    if name == "PWDB":
        z = np.load(os.path.join(DATA, "pwdb_cycles.npz"))
        X = z["X_rad"].astype(np.float32)
        T = _rd("pwdb_truth.csv")
        y = _col(T, "SBP_a")
        ok = np.isfinite(y) & (_col(T, "plausible") == 1)
        groups = np.arange(len(y))          # 每个虚拟受试者独立
    elif name == "VDB":
        z = np.load(os.path.join(DATA, "vitaldb_cycles.npz"), allow_pickle=True)
        X = z["X"].astype(np.float32)
        F = _rd("vitaldb_features.csv")
        y = _col(F, "sbp_wave")
        sid = np.asarray(z["subject_id"].tolist())
        uk, groups = np.unique(sid, return_inverse=True)
        ok = np.isfinite(y)
        # 分层变量：年龄/脉压/术前高血压是主轴，科室与心率为次轴，BMI/ASA 备用。
        # 注意 sex 在本表整列为空（VDB 导出的该字段未填），不能拿来分层。
        meta = dict(age=_col(F, "age"), htn=_col(F, "preop_htn"),
                    pp=_col(F, "pp_mmHg"), hr=_col(F, "hr"),
                    bmi=_col(F, "bmi"), asa=_col(F, "asa"),
                    rr_cv=_col(F, "rr_cv"), keep_frac=_col(F, "keep_frac"),
                    dept=np.array([r.get("department", "") for r in F], dtype=object))
    elif name == "MIMIC":
        z = np.load(os.path.join(DATA, "mimicbp_cycles.npz"))
        X = z["X"].astype(np.float32)
        y = _col(_rd("mimicbp_features.csv"), "sbp")
        uk, groups = np.unique(z["pid"].tolist(), return_inverse=True)
        ok = np.isfinite(y)
    elif name == "CM":
        # CM_VER 选择前处理版本：v1 原始片段 / v2 含 savgol 的旧链 / v3 现行链。
        # v3 只保留通过质量门的段（见 13 号），质量表随版本走，不能混用——
        # 用 v3 的波形配 v2 的门，会让被剔除的段重新混进来。
        # 默认必须取 v3。v1（real_cycles.npz）由 extract_real.py 生成，带通只到 4 Hz，
        # 切迹与高阶谐波在物理上已被削光；v3（real_cycles_v3.npz）才是 13 号的现行链。
        # 实测（audit_cm_person_level，人级嵌套 LOOCV，56 人）：
        #     v1  h1 -> SBP  ρ = +0.118 [−0.178, +0.374]   （不显著，4 Hz 带通毁掉信息）
        #     v3  h1 -> SBP  ρ = +0.389 [+0.138, +0.609]   （显著）
        #     v1  ridge29    ρ = −0.514 ；v3 ridge29 ρ = +0.072
        # 12/14 号在 v2.2 重跑时未设 CM_VER，CM 侧走的是 v1，故其 CM 数字全部偏低。
        # 保留环境变量是为了复现历史，不再作为默认。
        ver = os.environ.get("CM_VER") or "v3"
        cyc = {"v1": "real_cycles.npz", "v2": "real_cycles_v2.npz",
               "v3": "real_cycles_v3.npz"}[ver]
        qua = {"v2": "cm_quality.csv", "v3": "cm_quality_v3.csv"}.get(ver)
        z = np.load(os.path.join(DATA, cyc))
        X = z["X"].astype(np.float32)
        sid = z["sid"].tolist()
        seg = z["session"].tolist()
        if qua:
            q = {(r["subject_id"], r["segment"]): r["usable"] == "1"
                 for r in _rd(qua)}
            m = np.array([q.get((a, b), False) for a, b in zip(sid, seg)])
            X, sid, seg = X[m], [s for s, k in zip(sid, m) if k], [s for s, k in zip(seg, m) if k]
        lut = {}
        for r in _rd("labels_master.csv"):
            v = r.get("sbp", "")
            try:
                lut[r["subject_id"]] = float(v)
            except (TypeError, ValueError):
                pass
        y = np.array([lut.get(s, np.nan) for s in sid], np.float64)
        uk, groups = np.unique(np.asarray(sid), return_inverse=True)
        ok = np.isfinite(y)
        # sid 随 meta 走同一 ok 掩码：下游需要"段 -> 被试"映射（如 age 匹配）时
        # 必须用这里给出的对齐结果，不允许自行重算（G42 教训：错位=标签错配）。
        meta["sid"] = np.array(sid, dtype=object)
        # 版本必须随结果落盘：v1/v3 的段数与分布都不同，不记版本的结果无法复现，
        # 这正是 12/14 号 CM 数字偏差的来源。
        meta["cm_ver"] = np.array([ver] * len(y), dtype=object)
    else:
        raise KeyError(name)

    X, y, groups = X[ok], y[ok], groups[ok]
    meta = {k: v[ok] for k, v in meta.items()}
    # 质量门：平坦周期（峰谷差为 0）在归一化后会让 q=(p-pmin)/pp 除零，
    # 后续被中位数填补成伪观测。必须在此剔除，不能留给下游消化。
    pp = X.max(1) - X.min(1)
    flat = int((pp < 1e-6).sum())
    keep = pp >= 1e-6
    X, y, groups = X[keep], y[keep], groups[keep]
    meta = {k: v[keep] for k, v in meta.items()}
    groups = np.unique(groups, return_inverse=True)[1]
    X = (X - X.min(1, keepdims=True)) / (X.max(1, keepdims=True)
                                        - X.min(1, keepdims=True) + 1e-9)
    V, qn, dropped, aux = PL.build_l1(X, keep_nan=keep_nan, with_aux=True)
    return dict(X=X, y=y, groups=groups, V=V, qnames=qn, dropped=dropped,
                aux=aux, n_group=int(len(np.unique(groups))), flat=flat, meta=meta)


# ---------------------------------------------------------------- 训练
def grouped_oof(arch, X, y, groups, k=5, seed=0, epochs=80):
    gkf = GroupKFold(n_splits=min(k, len(np.unique(groups))))
    pred = np.full(len(y), np.nan, np.float32)
    for tr, te in gkf.split(X, y, groups):
        _, o = fit_model(arch, X[tr], y[tr], X[te], epochs=epochs, seed=seed)
        pred[te] = o["pred"]
    assert np.isfinite(pred).all(), "OOF 有未覆盖行"
    return pred


def ridge_oof(V, y, groups, cols=None, k=5):
    idx = np.arange(V.shape[1]) if cols is None else cols
    A = V[:, idx]
    gkf = GroupKFold(n_splits=min(k, len(np.unique(groups))))
    out = np.full(len(y), np.nan, np.float32)
    for tr, te in gkf.split(A, y, groups):
        out[te] = make_pipeline(StandardScaler(), Ridge(1.0)).fit(
            A[tr], y[tr]).predict(A[te])
    return out


# ---------------------------------------------------------------- 归因
def head_gain(model, sae, st):
    """每个 SAE 特征的单位激活对预测的贡献 c_f（解析）。

    输出头在表征上是线性的，表征又由解码器线性重建，故
        yhat - yhat0 = sum_f c_f * a_f ,  c_f = (w_head * sd) . dec_col_f
    与"置零再前向"的经验值应完全相等，本文件两者都算，用相对误差做自检。
    """
    w = model.head.weight.detach().squeeze(0)          # (64,)
    dec = sae.post.weight.detach()                     # (64, F)
    return (w * st["sd"].squeeze(0)) @ dec              # (F,)


def cluster_dims(V, names):
    """【v2.2 弃用，仅 audit_missing.py 等历史脚本仍在调用】簇的第一主成分。

    已被 ortho_dims 取代：17 号 §4 证明这套靶子没有定位能力（阳性对照 z=+0.42），
    且 8 个簇 PC1 的有效独立维度只有 1.62–3.32。保留此函数只为让历史审计脚本
    还能跑出当年的数字，新代码一律不得调用。

    为什么必须这样做
    ----------------
    直接对 35 个物理量的 rho^2 求和来分摊权重，会被"簇里有几列"决定：弦簇 12 列、滑簇 5 列，
    哪怕每列的解释力一样，弦簇也会自动拿到 71% 的份额。实测确实出现过"弦 65%"这种数字，
    它反映的是我在表里写了几行，不是模型在想什么。

    取 PC1 后每个簇只贡献一个数，份额与列的多少无关；共线的量本来就该塌成一个维度
    （PWDB 单周期内在维度只有 1.98，见 05 号），所以这一步同时解决了共线性。
    """
    P, cs = [], []
    for c in PL.TABLE:
        idx = [k for k, n in enumerate(names) if PL.cluster_of(n) == c]
        if not idx:
            continue
        A = V[:, idx]
        A = A - A.mean(0)
        A = A / (A.std(0) + 1e-12)
        u, sv, vt = np.linalg.svd(A, full_matrices=False)
        p = u[:, 0] * sv[0]
        if vt[0][int(np.argmax(np.abs(vt[0])))] < 0:   # 固化符号，保证可复现
            p = -p
        P.append(p)
        cs.append(c)
    return np.array(P).T.astype(np.float32), cs


class ShapleyAttrib:
    """按簇做 Shapley R^2 分解（LMG 法），作为 v2.2 的归因口径。

    为什么换掉"簇 PC1 / 正交主成分"两条路
    --------------------------------------
    两条路都在阳性对照上失败，失败的方式不同但根因相同：

      簇 PC1（v2.1）   没有定位能力：86.7% 的解释力植入单簇，机理份额只动 4.4pp
                      （z=+0.42），因为 rho^2 会摊给相关 0.7–0.9 的邻居。
      正交 PC（v2.2a） K=3 只覆盖 69–84% 的 L1 方差，落在子空间外的真实信号被
                      读成"未检出"。植入噪声方向时设备 z 只有 −0.3（v2.1 反能到 +3.1，
                      因为它的靶子里就有一个噪声 PC1）。

    而且这两条路都要先选一个基底（哪些维度、几个维度），而基底的选择会改变份额。
    Shapley 不需要基底：对每个特征只用"用簇子集 S 回归时的 R^2"这一个可测量 v(S)，
    再对所有加入顺序取平均边际增量。它精确可加（Σ_c φ_c = R^2_full），对共线簇
    给的是公平分账（这正是 LMG 被发明出来要解决的问题），且 1−R^2 直接就是
    "L1 解释不了的部分"。

    覆盖度与构成必须分开
    --------------------
    coverage = Σ_f d_f R^2_f / Σ_f d_f          模型的因果权重有多少能被 L1 解释
    comp_c   = Σ_f d_f φ_fc / Σ_f d_f R^2_f     可解释的部分落在哪个簇（和为 1）

    第一版把 φ 直接除以 Σd_f，于是份额之和等于 R^2；置换后 R^2→0，一切份额一起
    趋于 0，z 被推到 1e12 —— 那个统计量测的是"有没有 L1 关联"，不是"偏好哪一簇"。
    要看偏好，必须条件在 R^2>0 上取构成。

    能测什么、不能测什么（阳性对照实测，见 18 号 §4/§5）
    ------------------------------------------------
    能：设备/高频方向。植入噪声簇方向、覆盖度 82–96% 时，设备 z=+5.3…+6.5，
      重抽 100 次的标准差只有 0.01–0.05 —— 这根轴是活的。
    不能：机制簇之间的定位。植入 stiffness 方向 -> 机理 z 仅 +0.02…+0.16。
      原因不是方法弱，是数据本身：由 stiffness 四列线性组合出来的信号，用其余
      25 列也能几乎同样好地预测，Shapley 只能平分。这与 17 号"reflection~dicrotic
      r=+1.00"是同一件事的两种测法。
    半能：族构成的 z 有结构偏移（18 号 §5），机理轴被系统性压低 0.4–1.2σ。
      因此判定只能建在**设备轴上**（偏移小、效应大），不能建在"机理 vs 重述"的
      排序上（偏移与效应同量级）。族级结论一律以特征自助法的区间为准，不以 z 为准。
    """

    def __init__(self, V, names):
        self.cs = list(PL.TABLE)
        k = len(self.cs)
        self.subs = [S for r in range(k + 1)
                     for S in itertools.combinations(range(k), r)]
        self.wgt = {}
        for c in range(k):
            for S in self.subs:
                if c not in S:
                    s = len(S)
                    self.wgt[(c, S)] = (math.factorial(s) * math.factorial(k - s - 1)
                                        / math.factorial(k))
        self.Z = (V - V.mean(0)) / (V.std(0) + 1e-12)
        self.n = len(self.Z)
        self.names = list(names)
        self.labels = [PL.cluster_of(x) for x in names]
        self._setup(self.labels)

    def _setup(self, labels):
        C = self.Z.T @ self.Z / self.n
        self.rows, self.inv = {}, {}
        for S in self.subs:
            cols = [j for c in S for j, nm in enumerate(labels) if nm == self.cs[c]]
            self.rows[S] = np.array(cols, int)
            if cols:
                A = C[np.ix_(cols, cols)]
                self.inv[S] = np.linalg.pinv(A + 1e-9 * np.eye(len(cols)))

    def phi(self, G):
        """G=(p,F)=Z'Az/n -> phi (C,F) 与 R^2 (F,)；Σ_c φ_c = R^2 精确成立。"""
        r = {}
        F = G.shape[1]
        for S in self.subs:
            rows = self.rows[S]
            if len(rows) == 0:
                r[S] = np.zeros(F)
                continue
            g = G[rows]
            r[S] = np.einsum("if,if->f", g, self.inv[S] @ g)
        phi = np.zeros((len(self.cs), F))
        for c in range(len(self.cs)):
            for S in self.subs:
                if c in S:
                    continue
                phi[c] += self.wgt[(c, S)] * (r[tuple(sorted(S + (c,)))] - r[S])
        return np.clip(phi, 0, None), r[tuple(range(len(self.cs)))]

    def compose(self, G, d):
        phi, r2 = self.phi(G)
        w = d / max(d.sum(), 1e-12)
        cov = float((w * r2).sum())
        comp = (phi * w).sum(1) / max(cov, 1e-12)
        return phi, cov, comp


def attr_null(SD, G, d, nperm=200, seed=0):
    """簇标签置换零分布：保持每个簇的物理量个数，随机重排"哪几列算一个簇"。

    对"构成"这个问题，样本行置换是不对的 —— 它会切断特征与物理量的全部对应，
    使 R^2 趋近 0，所有份额一起塌到 0，测出来的是"有没有关联"。
    正确的零假设是：把 30 个测量按同样大小的块重新分组，模型的载荷还会显得
    这么偏某个族吗？因此置换的是标签，不是样本。
    """
    sizes = [len([1 for x in SD.labels if x == c]) for c in SD.cs]
    rng = np.random.default_rng(20261103 + seed)
    base = np.array(SD.labels, object)
    fams = [PL.family_of(c) for c in SD.cs]
    fc = np.zeros((nperm, len(SD.cs)))
    ff = np.zeros((nperm, 3))
    for b in range(nperm):
        perm = rng.permutation(len(SD.names))
        lab = np.empty(len(SD.names), object)
        done = 0
        for c, s in zip(SD.cs, sizes):
            lab[perm[done:done + s]] = c
            done += s
        SD._setup(list(lab))
        _, _, cp = SD.compose(G, d)
        fc[b] = cp
        for i, f in enumerate(("mechanism", "descriptor", "device")):
            ff[b, i] = sum(cp[j] for j, g in enumerate(fams) if g == f)
    SD._setup(base.tolist())
    return fc, ff


def attribute_block(model, Z, V, qn, y_raw, src, arch, seed=0, nperm=200):
    """表征 -> SAE -> 归因（Shapley R^2 按簇分账）-> 构成 + 覆盖度 + 置换 z。

    单源训练与混训共用这一段：两者只差"模型是怎么训出来的"，解剖手续必须一样。
    """
    Zt = torch.tensor(Z, device=DEV)
    sae, st = train_sae(Zt)
    with torch.no_grad():
        A_t = sae.encode((Zt - st["mu"]) / st["sd"])
        A = A_t.cpu().numpy()
        base = model.head(sae.post(A_t) * st["sd"] + st["mu"]).squeeze(-1)

    c = head_gain(model, sae, st).cpu().numpy()          # (F,)
    a_mean = A.mean(0)
    d_ana = np.abs(c * a_mean)

    # 自检：解析增益与置零经验值必须一致（抽查前 32 个特征，全跑太慢）
    d_emp = np.zeros(32, np.float32)
    for j in range(32):
        with torch.no_grad():
            a2 = A_t.clone(); a2[:, j] = 0.0
            z2 = sae.post(a2) * st["sd"] + st["mu"]
            d_emp[j] = float((model.head(z2).squeeze(-1) - base).abs().mean())
    ident_err = float(np.abs(d_ana[:32] - d_emp).sum() / (np.abs(d_emp).sum() + 1e-12))

    sd = ShapleyAttrib(V, qn)
    Az = (A - A.mean(0)) / (A.std(0) + 1e-12)
    G = sd.Z.T @ Az / len(A)
    phi, cov, comp = sd.compose(G, d_ana)
    fc, ff = attr_null(sd, G, d_ana, nperm=nperm, seed=seed)

    cs = sd.cs
    cl_share = {k: float(comp[i]) for i, k in enumerate(cs)}
    cw = {f: 0.0 for f in ("mechanism", "descriptor", "device")}
    for i, k in enumerate(cs):
        cw[PL.family_of(k)] = cw.get(PL.family_of(k), 0.0) + cl_share[k]
    fam_names = ("mechanism", "descriptor", "device")
    cl_z = {}
    for i, k in enumerate(cs):
        s = float(fc[:, i].std())
        cl_z[k] = float((cl_share[k] - fc[:, i].mean()) / (s + 1e-12))
    fam_z = {}
    for i, f in enumerate(fam_names):
        s = float(ff[:, i].std())
        fam_z[f] = float((cw[f] - ff[:, i].mean()) / (s + 1e-12))

    # 方向一致性：用簇的合成得分（簇内标准化量的均值）代替"唯一最优物理量"
    zc = np.zeros((len(V), len(cs)))
    for i, k in enumerate(cs):
        idx = [j for j, n in enumerate(qn) if PL.cluster_of(n) == k]
        zc[:, i] = sd.Z[:, idx].mean(1)
    rho_c = assoc(A, zc)                                  # (F,C)
    beta_c = np.array([spearmanr(zc[:, i], y_raw).statistic for i in range(len(cs))])
    rhoq = assoc(A, V)                                    # (F,p) 仅用于命名

    rows = []
    phi_sum = phi.sum(0)
    for f in range(sae.n_feat):
        if phi_sum[f] <= 1e-12 or a_mean[f] <= 0:
            rows.append(dict(source=src, arch=arch, seed=seed, feat=f, act=0.0,
                             cluster="dead", top_q="", rho_q="", second_q="",
                             rho_2="", w_cluster="", named="no", d_pred=0.0,
                             msc="", sign_c=0))
            continue
        j_top = int(np.argmax(phi[:, f]))
        cl_top = cs[j_top]
        sub = [k for k, n in enumerate(qn) if PL.cluster_of(n) == cl_top]
        rq = np.nan_to_num(rhoq[f], nan=0.0)
        j = max(sub, key=lambda k: abs(rq[k]))
        tq, rho_t = qn[j], float(rq[j])
        _, _, sec, rho_s, _ = PL.top_two(rq, qn)
        ok = (np.sign(c[f]) * np.sign(np.nan_to_num(rho_c[f, j_top]))
              == np.sign(beta_c[j_top]))
        rows.append(dict(source=src, arch=arch, seed=seed, feat=f,
                         act=round(float(a_mean[f]), 5),
                         cluster=cl_top, top_q=tq, rho_q=round(rho_t, 3),
                         second_q=sec,
                         rho_2="" if not np.isfinite(rho_s) else round(rho_s, 3),
                         w_cluster=round(float(phi[j_top, f] / phi_sum[f]), 3),
                         named="yes" if phi[j_top, f] / phi_sum[f] >= 0.35 else "no",
                         d_pred=round(float(d_ana[f]), 5),
                         msc=1.0 if ok else 0.0,
                         sign_c=int(np.sign(c[f]))))

    alive = [r for r in rows if r["cluster"] != "dead"]
    dw = np.array([r["d_pred"] for r in alive])
    ms = np.array([r["msc"] for r in alive], float)
    msc_rate = float((dw * ms).sum() / max(dw.sum(), 1e-12))
    thr = np.quantile(dw, 0.66) if len(dw) else 0
    key = dw >= thr
    msc_key = float((dw[key] * ms[key]).sum() / max(dw[key].sum(), 1e-12))
    cl_msc = {}
    for i, k in enumerate(cs):
        m = np.array([r["cluster"] == k for r in alive])
        cl_msc[k] = float((dw[m] * ms[m]).sum() / dw[m].sum()) if m.sum() else np.nan

    return dict(rows=rows, clusters=cs, cl_share=cl_share, cw=cw, fam_z=fam_z,
                cl_z=cl_z, coverage=cov, unattributed=1.0 - cov, nperm=int(nperm),
                amb=float(np.mean([r["w_cluster"] for r in alive])) if alive else np.nan,
                msc=msc_rate, msc_key=msc_key, cl_msc=cl_msc,
                alive=len(alive),
                named=sum(1 for r in alive if r["named"] == "yes"),
                ident_err=ident_err, sae_r2=st["r2"],
                A=A, c=c, rhoq=rhoq, beta=beta_c)


def verdict_of(rho, cw, fam_z=None, coverage=np.nan):
    """判定（v2.2）：只用阳性对照证明**可测**的那一根轴。

    旧口径（v2.1 及以前）用 "机理份额 >= 0.60" 判 A。17 号证明这个门槛没有推断
    内容：机理族占 8 个簇里的 5 个，随机期望就是 5/8 = 62.5%，实测 52.7–64.1% 正
    落在基线附近，四源 z 全部 <= +0.38。

    v2.2 改为按置换零分布的 z 判，并且**主动放弃"机理 vs 重述"的排序**——
    18 号 §4 的阳性对照显示它测不出来：植入纯 stiffness 方向，机理 z 只有 +0.02…+0.16。
    原因在数据：由 stiffness 四列构成的信号，用其余 25 列也能几乎同样好地预测。
    只有设备/高频轴是活的（植入噪声方向 -> 设备 z=+5.3…+6.5，覆盖度 82–96%）。

    阈值为什么是 2.6 而不是 2.0
    ---------------------------
    18 号 §5 把零分布本身测了出来（每源 200 次独立重抽 A⊥V）：
    名义 |z|>2 的误报率实测 6–21%，且存在系统性位置偏移（机理 −1.18…−0.44、
    重述 +0.10…+0.63、设备 +0.13…+0.64），偏移来自"真实分簇比同大小的随机分组
    更内部相关"这件结构性事实，与 d 的集中度无关。按实测 μ/σ 反推单侧 5% 分位，
    设备轴的门槛应落在 −1.6…−2.2，取 2.6 约为单侧 1%，比旧值更保守。
    实测八配置的 device_z 全部落在 (−2.1, +1.4)，故抬高阈值**不改变任何一条判定**。

      A 读物理量   rho >= 0.35 且 设备 z < -2.6 且 coverage >= 0.30
      B 拟合噪声   设备 z > +2.6
      C 没学到     rho < 0.35
      D 未检出     其余 —— 只能说"没有证据"，不写成"它在重述"

    名字从"读机制"改成"读物理量"是必须的：这条判据只支持"不靠设备噪声、且因果
    权重确实落在 L1 上"，不支持"落在机制量而不是重述量上"。后者要等数据能分开
    重搏/反射/时程之后才谈得上（见 phys_layer 的 GAPS）。
    """
    if fam_z is None:
        fam_z = dict.fromkeys(("mechanism", "descriptor", "device"), 0.0)
    if rho < THR_RHO:
        return "C 没学到"
    if fam_z["device"] > THR_Z:
        return "B 拟合噪声"
    if fam_z["device"] < -THR_Z and np.isfinite(coverage) and coverage >= THR_COV:
        return "A 读物理量"
    return "D 未检出"


def analyse(arch, src, d, epochs=80, seed=0, nperm=200):
    X, y_raw, groups, V, qn = d["X"], d["y"], d["groups"], d["V"], d["qnames"]
    y = (y_raw - y_raw.mean()) / y_raw.std()

    pred = grouped_oof(arch, X, y, groups, seed=seed, epochs=epochs)
    rho = float(spearmanr(pred, y).statistic)
    r2 = float(1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())

    # 受试者级 rho：同一个人多段时，逐段 rho 会被"同人同标签"抬高
    if d["n_group"] < len(y) * 0.9:
        py = np.zeros(d["n_group"]); yy = np.zeros(d["n_group"])
        for g in range(d["n_group"]):
            m = groups == g
            py[g] = pred[m].mean(); yy[g] = y_raw[m].mean()
        rho_subj = float(spearmanr(py, yy).statistic)
    else:
        rho_subj = np.nan

    model, out = fit_model(arch, X, y, X, epochs=epochs, seed=seed)
    B = attribute_block(model, out["Z"], V, qn, y_raw, src, arch, seed=seed,
                        nperm=nperm)
    cw = B["cw"]
    fz = B["fam_z"]

    rho_l1 = float(spearmanr(ridge_oof(V, y, groups), y).statistic)
    nz = [i for i, n in enumerate(qn) if PL.cluster_of(n) == "noise"]
    rho_nz = float(spearmanr(ridge_oof(V, y, groups, nz), y).statistic)
    verdict = verdict_of(rho, cw, fz, B["coverage"])

    print(f"[{src:<5s}/{arch:<11s}] n={len(y):>5d} 组={d['n_group']:>4d} "
          f"rho={rho:.3f} (人级 {rho_subj:.3f}) 覆盖={B['coverage']*100:.0f}% "
          f"机制={cw['mechanism']*100:.0f}%(z{fz['mechanism']:+.1f}) "
          f"重述={cw['descriptor']*100:.0f}%(z{fz['descriptor']:+.1f}) "
          f"噪声={cw['device']*100:.0f}%(z{fz['device']:+.1f}) "
          f"MSC={B['msc']:.2f}/{B['msc_key']:.2f} -> {verdict}",
          flush=True)

    return dict(source=src, arch=arch, n=len(y), n_group=d["n_group"],
                rho=rho, rho_subj=rho_subj, r2=r2,
                sae_r2=B["sae_r2"], alive=B["alive"], named=B["named"],
                mech=cw["mechanism"], desc=cw["descriptor"], device=cw["device"],
                mech_z=fz["mechanism"], desc_z=fz["descriptor"],
                device_z=fz["device"], coverage=B["coverage"],
                unattributed=B["unattributed"], nperm=B["nperm"],
                cl_msc=B["cl_msc"], cl_z=B["cl_z"], seed=seed,
                ambig=B["amb"], msc=B["msc"], msc_key=B["msc_key"],
                rho_l1=rho_l1, rho_noise=rho_nz,
                ident_err=B["ident_err"], verdict=verdict,
                cl_share=B["cl_share"], rows=B["rows"], A=B["A"], V=V, qn=qn, y=y,
                c=B["c"], rhoq=B["rhoq"], clusters=B["clusters"], beta=B["beta"])


def run_pooled(sources, arch, cache, epochs=80, seed=0, k=5, nperm=200):
    """混训对照：四源合成一个模型，再按源各自解剖。

    目的只有一个 —— 检验"各训一个"这个主张。09 号用"215/256 个特征只对单库有反应"
    间接论证过混训会把表征撕成按源分块；这里把它量化成同一把尺子下的份额变化：
    若混训后 device（高频噪声）份额显著上升，说明模型部分地在靠"认库"而不是"读波"。

    标签处理：各源先各自 z 化再拼接，否则 MIMIC（中位 111）会压掉 CM（中位 137）。
    组编号按源加偏移，避免不同源的 id 撞车导致受试者泄漏。
    """
    Xs, ys, gs, src_of = [], [], [], []
    off = 0
    for i, s in enumerate(sources):
        d = cache[s]
        Xs.append(d["X"])
        ys.append((d["y"] - d["y"].mean()) / d["y"].std())
        gs.append(d["groups"] + off)
        src_of.append(np.full(len(d["y"]), i))
        off += int(d["groups"].max()) + 1
    X = np.concatenate(Xs); y = np.concatenate(ys)
    groups = np.concatenate(gs); src_of = np.concatenate(src_of)

    print(f"[pooled/{arch}] n={len(y)} 组={len(np.unique(groups))} "
          f"源={len(sources)}", flush=True)
    pred = grouped_oof(arch, X, y, groups, k=k, seed=seed, epochs=epochs)
    model, out = fit_model(arch, X, y, X, epochs=epochs, seed=seed)
    Z = out["Z"]

    res = []
    for i, s in enumerate(sources):
        m = src_of == i
        d = cache[s]
        B = attribute_block(model, Z[m], d["V"], d["qnames"], d["y"], s, arch,
                            seed=seed, nperm=nperm)
        rho = float(spearmanr(pred[m], y[m]).statistic)
        cw = B["cw"]
        fz = B["fam_z"]
        res.append(dict(source=s, arch=arch, arm="pooled", n=int(m.sum()), rho=rho,
                        mech=cw["mechanism"], desc=cw["descriptor"],
                        device=cw["device"], mech_z=fz["mechanism"],
                        desc_z=fz["descriptor"], device_z=fz["device"],
                        coverage=B["coverage"], unattributed=B["unattributed"],
                        msc=B["msc"], msc_key=B["msc_key"],
                        ambig=B["amb"], alive=B["alive"], named=B["named"],
                        sae_r2=B["sae_r2"],
                        verdict=verdict_of(rho, cw, fz, B["coverage"]),
                        cl_share=B["cl_share"], cl_z=B["cl_z"], rows=B["rows"],
                        clusters=B["clusters"]))
        print(f"[pooled/{arch} -> {s:<5s}] rho={rho:.3f} 机制={cw['mechanism']*100:.0f}% "
              f"重述={cw['descriptor']*100:.0f}% 噪声={cw['device']*100:.0f}% "
              f"MSC={B['msc']:.2f} -> {res[-1]['verdict']}", flush=True)
    return res


def write_pooled(pres):
    prow = [r for R in pres for r in R["rows"]]
    pfn = os.path.join(DATA, "probe_mediator_pooled.csv")
    with open(pfn, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(prow[0]))
        w.writeheader(); w.writerows(prow)
    pmeta = []
    for R in pres:
        for k in ("n", "rho", "mech", "desc", "device", "mech_z", "desc_z",
                  "device_z", "coverage", "unattributed", "msc", "msc_key",
                  "ambig", "alive", "named", "sae_r2", "verdict"):
            pmeta.append(dict(source=R["source"], arch=R["arch"], arm="pooled",
                              key=k, value=R[k]))
        for k, v in R["cl_share"].items():
            pmeta.append(dict(source=R["source"], arch=R["arch"], arm="pooled",
                              key=f"share_{k}", value=round(v, 4)))
        for k, v in R["cl_z"].items():
            pmeta.append(dict(source=R["source"], arch=R["arch"], arm="pooled",
                              key=f"clz_{k}", value=round(v, 3)))
    with open(os.path.join(DATA, "probe_mediator_pooled_meta.csv"), "w",
              newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["source", "arch", "arm", "key", "value"])
        w.writeheader(); w.writerows(pmeta)
    print("\n混训对照（各源 z 化后合并训练，再按源解剖）")
    print(f"{'源':<6s}{'架构':<12s}{'rho':>8s}{'机制%':>8s}{'重述%':>8s}"
          f"{'噪声%':>8s}{'MSC':>7s}{'判定':>12s}")
    for R in pres:
        print(f"{R['source']:<6s}{R['arch']:<12s}{R['rho']:>8.3f}"
              f"{100*R['mech']:>8.1f}{100*R['desc']:>8.1f}"
              f"{100*R['device']:>8.1f}{R['msc']:>7.2f}{R['verdict']:>12s}")
    print(f"-> {pfn}")


def l1_table(src, d):
    """每个物理量与血压的边际关系：这是 L2（中医概念）唯一允许引用的证据层。"""
    y = d["y"]
    out = []
    for j, n in enumerate(d["qnames"]):
        out.append(dict(source=src, quantity=n, cluster=PL.cluster_of(n),
                        rho=round(float(spearmanr(d["V"][:, j], y).statistic), 3)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=",".join(SRC))
    ap.add_argument("--archs", default="mlp,transformer")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--pooled", action="store_true",
                    help="追加混训对照：四源合一个模型，看噪声份额是否被抬高")
    ap.add_argument("--pooled-only", dest="pooled_only", action="store_true",
                    help="只跑混训对照，跳过单源实验")
    ap.add_argument("--nperm", type=int, default=200,
                    help="置换零分布次数；0 表示跳过（只用于快速冒烟测试）")
    ap.add_argument("--out", default="probe_mediator.csv")
    a = ap.parse_args()

    print(f"[env] device={DEV} 中介层 hash={PL.check_frozen()}")
    srcs = [s for s in a.sources.split(",") if s]
    archs = [s for s in a.archs.split(",") if s]
    seeds = [int(s) for s in a.seeds.split(",") if s]

    cache = {s: load_source(s) for s in srcs}
    for s in srcs:
        d = cache[s]
        print(f"[load] {s:<5s} n={len(d['y']):>5d} 组={d['n_group']:>4d} "
              f"L1={d['V'].shape[1]} 平坦剔除={d['flat']} 丢弃={d['dropped']} "
              f"SBP={d['y'].min():.0f}-{d['y'].max():.0f}")

    if a.pooled_only:
        pres = []
        for ar in archs:
            pres += run_pooled(srcs, ar, cache, epochs=a.epochs, seed=seeds[0],
                               nperm=a.nperm)
        write_pooled(pres)
        return

    res, l1rows = [], []
    for s in srcs:
        l1rows += l1_table(s, cache[s])
        for ar in archs:
            for sd in seeds:
                res.append(analyse(ar, s, cache[s], epochs=a.epochs, seed=sd,
                                   nperm=a.nperm))

    rows = [r for R in res for r in R["rows"]]
    fn = os.path.join(DATA, a.out)
    stem = os.path.splitext(a.out)[0]
    with open(fn, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    if a.pooled or a.pooled_only:
        pres = []
        for ar in archs:
            pres += run_pooled(srcs, ar, cache, epochs=a.epochs, seed=seeds[0],
                               nperm=a.nperm)
        write_pooled(pres)

    l1fn = os.path.join(DATA, f"{stem}_l1.csv")
    with open(l1fn, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(l1rows[0]))
        w.writeheader(); w.writerows(l1rows)

    np.savez(os.path.join(DATA, f"{stem}.npz"),
             **{f"{R['source']}_{R['arch']}_s{R['seed']}_{k}": R[k]
                for R in res for k in ("A", "V", "c", "rhoq", "y")},
             qnames=np.array(res[0]["qn"]),
             clusters=np.array(res[0]["clusters"]))

    meta = []
    for R in res:
        for k, v in R.items():
            if k in ("rows", "A", "V", "qn", "y", "c", "rhoq", "clusters",
                     "beta", "cl_share", "cl_msc", "cl_z", "seed"):
                continue
            meta.append(dict(source=R["source"], arch=R["arch"], key=k,
                             value="" if (isinstance(v, float) and np.isnan(v)) else v))
        for k, v in R["cl_share"].items():
            meta.append(dict(source=R["source"], arch=R["arch"],
                             key=f"share_{k}", value=round(v, 4)))
        for k, v in R["cl_msc"].items():
            meta.append(dict(source=R["source"], arch=R["arch"],
                             key=f"clmsc_{k}",
                             value="" if not np.isfinite(v) else round(v, 4)))
        for k, v in R["cl_z"].items():
            meta.append(dict(source=R["source"], arch=R["arch"],
                             key=f"clz_{k}", value=round(v, 3)))
    # CM 必须记下用的是哪一版预处理：v1（4 Hz 带通）与 v3（10 Hz）的人级 h1-SBP
    # 相关是 0.118 与 0.389，不记版本的结果无法复现也无法追责。
    for R in res:
        if R["source"] == "CM":
            meta.append(dict(source="CM", arch=R["arch"], key="cm_ver",
                             value=os.environ.get("CM_VER") or "v3"))
    with open(os.path.join(DATA, f"{stem}_meta.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["source", "arch", "key", "value"])
        w.writeheader(); w.writerows(meta)

    print("\n" + "=" * 108)
    print(f"{'源':<6s}{'架构':<12s}{'n':>6s}{'组':>5s}{'OOF rho':>9s}{'人级rho':>9s}"
          f"{'覆盖%':>6s}{'机制%':>7s}{'z':>7s}{'重述%':>7s}{'z':>7s}"
          f"{'噪声%':>7s}{'z':>7s}{'MSC':>6s}{'判定':>10s}")
    for R in res:
        print(f"{R['source']:<6s}{R['arch']:<12s}{R['n']:>6d}{R['n_group']:>5d}"
              f"{R['rho']:>9.3f}{R['rho_subj']:>9.3f}{100*R['coverage']:>6.0f}"
              f"{100*R['mech']:>7.1f}{R['mech_z']:>+7.1f}"
              f"{100*R['desc']:>7.1f}{R['desc_z']:>+7.1f}"
              f"{100*R['device']:>7.1f}{R['device_z']:>+7.1f}"
              f"{R['msc']:>6.2f}{R['verdict']:>10s}")

    print("\n各簇因果份额（%），列 = 簇")
    cls = res[0]["clusters"]
    print(f"{'源/架构':<20s}" + "".join(f"{c[:8]:>9s}" for c in cls))
    for R in res:
        print(f"{R['source']+'/'+R['arch']:<20s}"
              + "".join(f"{100*R['cl_share'].get(c, 0):>9.1f}" for c in cls))

    print("\n基线（OOF rho）：L1-Ridge = 30 个物理量的信息上限；噪声簇-Ridge = 只给 9 个高频量")
    for R in {r['source']: r for r in res}.values():
        print(f"  {R['source']:<6s} 模型 {R['rho']:.3f} | L1上限 {R['rho_l1']:.3f} "
              f"| 仅噪声簇 {R['rho_noise']:+.3f} | 解析/经验恒等误差 {R['ident_err']:.2e}")
    print(f"\n-> {fn}\n-> {l1fn}")


if __name__ == "__main__":
    main()
