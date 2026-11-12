"""物理量中介层 L1 —— 把 256 点周期映射成一组可测物理量，并冻结分簇与概念映射。

三层结构
--------
L0 波形      256 点，逐周期 min-max（唯一跨设备可比口径，见 06/11 的 UNCAL 协议）
L1 物理量    从 L0 直接可测，四源共用 pwfeat.seg_feats 同一实现（v2.2 起 29 个）
L2 中医概念  只通过 L1 映射，**永不作为标签**

为什么需要这一层
----------------
脉象标注在文献里是"用手指的感觉定义"而非数值阈值，跨研究不可比；若把脉象当训练标签，
标签本身就是规则算出来的，SAE 拆出来的必然是那条规则 —— 循环论证（11 号 §4）。
物理量可测、可复现、四源同口径，因此拿它当中介层：模型照常学「波形 -> 血压」，
SAE 拆出的特征归因到 L1，再看权重落在哪个机制簇。这样"它在想什么"就有了可比靶子，
而"想得对不对"由符号一致性（见 probe_mediator.py 的 MSC）来判，不再依赖专家标注。

为什么 L1 只含无量纲量
----------------------
模型输入已逐周期 min-max，绝对幅度（pp_mmHg）被消灭成常数 1，不存在可归因的方差。
另外 VDB/MIMIC 是有创校准压，其波形纵轴本身就是 mmHg —— 若保留绝对幅度，"预测收缩压"
退化成"读波形最大值"，解剖无意义。故 L1 一律无量纲，这是本层最容易被违反的前提，
模块内以 assert 守住。

冻结
----
TABLE 一旦冻结即以 SHA256 记入 `data/phys_mediator_v1.json` 与规范文档，事后改动会直接
触发 RuntimeError。理由：簇的划法会改变份额百分比，若不冻结，任何结论都可以通过事后
调整分簇来制造。
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


import hashlib
import json
import os

import numpy as np

import pwfeat

ROOT = _repo.ROOT
DATA = os.path.join(ROOT, "data")
FREEZE_JSON = os.path.join(DATA, "phys_mediator_v1.json")

# FROZEN_SHA 由 --freeze 生成后钉死；空串表示尚未冻结（仅用于首次生成）
FROZEN_SHA = "f557d04dc5e431a9"
VERSION = "v2.3"

# v1 -> v2 变更记录（2026-11-01，首次贯通四源解剖时发现）
#   v1 把 8 个簇一律记为 family="phys"，导致"整体形态重述"与"机制"混为一谈：
#   kurt / h1 / mbp 这类分布描述量与血压的 |rho| 均在 0.78 上下，且彼此共线，
#   模型只要复述波形整体形态就会拿到 60% 的"生理"份额，被判成 A（读生理量）。
#   这与 11 号 §1 把 PP_amp/MBP 划为"压力表现"、不可作为解释，是同一类错误。
#   v2 改为三族：mechanism（有独立物理机制的形态/时相量）/ descriptor（波形重述）/
#   device（非生理）。簇与物理量的归属**一个都没有改**，只改家族的划法，
#   因此 v1 的簇内份额仍可与 v2 对照。
#
# v2.0 -> v2.1 变更记录（2026-11-02，见 15 号 §1 的 P0 三项）
#   两个物理量被逐出 L1（35 -> 33），原因是它们测的是**滤波后的平滑度**而非脉搏：
#     q_entropy      完整链 vs 仅高通 |Δ|=0.667，97% 的样本受平滑影响
#     upstroke_slope 完整链 vs 仅高通 |Δ|=0.331，同为平滑度代理
#   二者进入 L1 会让"设备噪声"份额虚高，因为它们把处理链的痕迹记成了生理量。
#   同时 p2_rel 的分母由 1 改为 q[i1]（见 pwfeat.seg_feats），解除 21.3% 的上溢饱和。
#   簇的划法与 family 归属未改；v2.0 的份额仍可与 v2.1 对照（剔除簇内列数变化已由
#   簇级 PC1 归因消化，见 12 号 §3）。

# v2.1 -> v2.2 变更记录（2026-11-03，见 17 号 §2/§3 的可辨识性审查）
#   17 号证明了两件事：(a) 33 个物理量里有多组精确重复/镜像，独立量只有约 22 个，
#   "33 个量里有几个同号"这种计数被重复项虚增；(b) 8 个簇的 PC1 有效独立维度只有
#   1.62–3.32（参与率），reflection 与 dicrotic 的 PC1 相关高达 +1.00，notch 是
#   p_notch_rel 的镜像 —— 三个簇测的是同一件事。于是：
#     1. 删除 4 个重复量（33 -> 29），依据是逐源实测（下表）：
#          ap_rel / aix_proxy  ≡ p2_rel - q[P1]   （PWDB/VDB/MIMIC/CM 全部 r=1.000000）
#          p_notch_rel         ≡ 1 - notch_depth   （PWDB/CM 精确，VDB/MIMIC r=-0.995/-0.992）
#          t2                  ≡ t_dicrotic        （PWDB/CM 逐点相同，VDB/MIMIC r=0.9995/0.9975）
#     2. reflection + notch + dicrotic 合并为一簇（8 -> 6 簇）。这三者在 L1 里无法
#        分离：P2 时刻就是重搏峰时刻（t2 ≡ t_dicrotic），切迹深度是反射强度的镜像。
#        合并后"弦/滑"的区分在 L1 层面失效，写进 GAPS，不再假装能分开。
#   份额口径同时改为正交主成分靶子（probe_mediator.ortho_dims），簇不再当靶子，
#   只作为"这一维的方差由谁贡献"的事后标签。因此 v2.2 的份额与 v2.0/v2.1 的
#   **不可直接相减**，两者是不同口径（17 号 §7 已说明旧口径是构成比）。

# ---------------------------------------------------------------- 冻结表
# 簇名 -> (含哪些物理量, 机制家族, 中医概念, 一句话物理含义)
# 归责原则：每个物理量只归属一个簇，保证份额可加；共线的量放同一簇，避免把
# "同一机制的两种写法"当成两个机制（11 号已证明物理量之间 |rho| 常 >0.9）。
TABLE = {
    "stiffness": dict(
        feats=["t_90", "t_50up", "h3_r", "h4_r"],
        family="mechanism", concept="弦", gloss="上升支陡度与高阶谐波比 = 血管张力/硬度"),
    "reflection": dict(
        feats=["p2_rel", "t_notch", "notch_depth", "sdptg_b"],
        family="mechanism", concept="弦",
        gloss="收缩后反射波/肩部强度与时相 = 早期波反射；v2.3 加 sdptg_b（二阶导 b/a），"
              "MIMIC 上与全部形态量秩相关 ≤0.25，是独立测量"),
    "dicrotic": dict(
        feats=["t_dicrotic", "sdptg_d"],
        family="mechanism", concept="滑",
        gloss="重搏波时相与二阶导 d/a = 晚期(舒张)波；v2.3 从 reflection 拆出。"
              "分半信度仅 MIMIC 成立（b 0.68 / d 0.54），VDB/PWDB 一人一段不可测，"
              "CM 0.22/0.09 = 噪声（19 号 §6）"),
    "duration": dict(
        feats=["t1", "t_50down", "width50"],
        family="mechanism", concept="滑", gloss="收缩峰时相与半高宽 = 波形圆滑/陡峭"),
    "shape": dict(
        feats=["skew", "kurt", "mbp_mmHg"],
        family="descriptor", concept="涩", gloss="幅值分布形状 = 波形的统计重述，非独立机制"),
    "spectrum": dict(
        feats=["h1", "h2", "h3", "h4", "h2_r"],
        family="descriptor", concept="通用", gloss="基频与前四阶谐波 = 整体频谱包络，非独立机制"),
    "noise": dict(
        feats=["h5", "h6", "h7", "h8", "h5_r", "h6_r", "h7_r", "h8_r", "hf_energy"],
        family="device", concept="无", gloss="5 阶以上高频能量 = 量化/噪声/采样率指纹，非生理"),
}

# 中医概念 -> (依赖的 L1 簇, 传统描述, 本层能否支撑)
CONCEPT = {
    "弦": dict(clusters=["stiffness", "reflection"],
              classic="端直以长，如按琴弦", ok=True,
              note="张力高、反射早而强 —— L1 有直接对应量"),
    "滑": dict(clusters=["duration", "dicrotic"],
              classic="往来流利，如珠走盘", ok=True,
              note="v2.3 起 dicrotic 独立成簇（t_dicrotic + sdptg_d），'滑'的重搏成分"
                   "与'弦'的反射成分可分。但测量信度仅 MIMIC 成立（ICC_d 0.54）；"
                   "VDB/PWDB 一人一段不可测，CM 0.09 = 噪声 —— 结论只能限定在这类源上"),
    "涩": dict(clusters=["shape"],
              classic="往来艰涩，如轻刀刮竹", ok=True,
              note="L1 只覆盖单周期内不规则；逐搏变异需 beat-level 数据，本层不覆盖"),
    "数/迟": dict(clusters=[], classic="一息五至以上为数，三至以下为迟", ok=False,
                note="需心率，时间轴归一化后被消掉 —— L1 缺口，须从 cycle_s 另补"),
    "洪/细": dict(clusters=[], classic="来盛去衰为洪，细直而软为细", ok=False,
                note="需绝对幅度，UNCAL 口径下不存在 —— L1 缺口，仅校准源可用"),
    "浮/沉": dict(clusters=[], classic="举之有余按之不足为浮", ok=False,
                note="需多通道压力梯度，仅 CM 有 25 通道 —— L1 缺口"),
}

# 论证前必须声明的缺口，写进报告，不许在正文里含糊过去
GAPS = ["数/迟（心率被时间归一化消掉）", "洪/细（绝对幅度被 min-max 消掉）",
        "浮/沉（需多通道，L1 为单通道）", "涩的逐搏变异（L1 为单周期）",
        "重搏簇的跨源可比性（v2.3 用 SDPTG d/a 在特征定义层拆开了重搏/反射，但"
        "实测 PWDB 上 sdptg_b~p2_rel r=+0.80（合成单峰，b 波必然跟随主峰）、"
        "VDB/PWDB 一人一段无法测信度、CM 0.09=噪声 —— dicrotic 簇的可测域只有"
        "MIMIC 这类多搏真双峰源；CM 采集层不可分是硬件限制，见 19 号 §3）"]

_CLUSTER_OF = {f: c for c, v in TABLE.items() for f in v["feats"]}
_FAMILY_OF = {c: v["family"] for c, v in TABLE.items()}

# seg_feats 仍会算出、但**明确不进 L1** 的量。列在这里是为了让 assert_cover 能区分
# "漏了分簇"（bug）与"有意剔除"（决策），否则删列会静默通过或者一律报错。
EXCLUDED_FEATS = {
    "q_entropy": "幅值分布熵=平滑度代理，|Δ|=0.667 随处理链漂移（15 号 §1）",
    "upstroke_slope": "上升支斜率=平滑度代理，|Δ|=0.331 随处理链漂移（15 号 §1）",
    "p2_found": "P2 检出的布尔标记，0/1 无生理量纲；用于下游过滤而非归因（16 号 §2）",
    # v2.2 新增：与保留量精确重复或近似镜像，留着会让"多少个量同号"这类计数虚增
    "aix_proxy": "≡ ap_rel，两者与 p2_rel 相差常数 q[P1]（四源 r=1.000000，17 号 §3）",
    "ap_rel": "≡ aix_proxy，同上；p2_rel 已有同一信息",
    "p_notch_rel": "≡ 1 − notch_depth（PWDB/CM 精确，VDB/MIMIC r=−0.995/−0.992）",
    "t2": "≡ t_dicrotic（PWDB/CM 逐点相同；VDB/MIMIC r=0.9995/0.9975）—— P2 即重搏峰",
    # v2.3 新增：第 5 个重复量，v2.2 去重时漏网（17 号只查了 ap_rel/aix_proxy/t2/p_notch_rel）
    "p_dicrotic_rel": "≡ p2_rel（PWDB max|diff|=0；MIMIC/VDB mean|diff|=0.006/0.001，"
                      "秩相关 +1.000）—— 检测器恒等式 i3≡i2 的产物，见 19 号 §7",
}


def cluster_of(feat):
    return _CLUSTER_OF.get(feat, "other")


def family_of(cluster):
    return _FAMILY_OF.get(cluster, "other")


def table_hash():
    blob = json.dumps(TABLE, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def check_frozen():
    got = table_hash()
    if FROZEN_SHA and got != FROZEN_SHA:
        raise RuntimeError(
            f"物理量中介层已被改动：table_hash={got} 冻结值={FROZEN_SHA}。"
            "任何分簇改动都必须先出新版本（v2）并在文档里说明，不许就地改。")
    return got


def assert_cover(names):
    """seg_feats 每个输出必须被某个簇接住或被显式剔除，否则份额不闭合。"""
    miss = [n for n in names if n not in _CLUSTER_OF and n not in EXCLUDED_FEATS]
    if miss:
        raise RuntimeError(f"L1 覆盖不全，以下物理量既未分簇也未剔除：{miss}")
    dup = len(_CLUSTER_OF) - len(set(_CLUSTER_OF))
    if dup:
        raise RuntimeError(f"有 {dup} 个物理量被重复分簇，份额会重复计数")
    both = [n for n in EXCLUDED_FEATS if n in _CLUSTER_OF]
    if both:
        raise RuntimeError(f"以下物理量同时被剔除又被分簇，语义冲突：{both}")


# ---------------------------------------------------------------- 计算
UNITLESS_DROP = ("pp_mmHg",)   # 归一化后恒等于 1，无方差可言


AUX_FEATS = ("p2_found",)   # 不进 L1，但要跟 V 逐行对齐，供下游做条件分析


def build_l1(X, keep_nan=False, with_aux=False):
    """X:(n,256) 逐周期 min-max 后的波形 -> (V, names[, dropped[, aux]])。

    直接复用 pwfeat.seg_feats（唯一合法实现，哈希锁定），不在这里另写一份，
    否则四源之间就不是同一个物理量了。

    keep 必须同时排除 UNITLESS_DROP（归一化后无方差）与 EXCLUDED_FEATS（已判定
    不是物理量）。只让 assert_cover 放行而不真删，会让 V 里留下 cluster="other"
    的列：簇级份额不受影响（归因只走 8 个 PC1），但量级命名口径会选到它们，
    且"L1 有几个物理量"这件事会说成 36 而不是 33。

    keep_nan
    --------
    默认 False：把缺失（P2 未检出导致的 p2_rel / ap_rel / aix_proxy）填成列中位数，
    因为归因（PCA / Ridge）要求稠密矩阵。
    True：原样保留 NaN，只给"逐量边际相关"这类本就能 pairwise 的统计用。
    两者不可混：一旦填补先发生，下游任何 `isfinite(x)` 掩码都会退化成恒真，
    于是"pairwise"只是名义上的（17 号 §3 就是踩了这个坑）。
    """
    feats = [pwfeat.seg_feats(x) for x in X]
    names = list(feats[0].keys())
    V = np.array([[f.get(k, np.nan) for k in names] for f in feats], np.float64)
    if not keep_nan:
        V = np.where(np.isfinite(V), V, np.nanmedian(V, 0))
    # 方差判据必须用 nanstd：含缺失的列用 np.std 得到 nan，而 nan > 1e-9 为 False，
    # 整列会被当成常数删掉 —— 那是把"有缺失"读成"无方差"，会把 reflection 簇
    # 直接砍掉一半。这里显式分开这两种语义。
    sd = np.nanstd(V, 0)
    keep = [j for j, n in enumerate(names)
            if n not in UNITLESS_DROP and n not in EXCLUDED_FEATS
            and np.isfinite(sd[j]) and sd[j] > 1e-9]
    dropped = [n for j, n in enumerate(names) if j not in keep]
    names = [names[j] for j in keep]
    V = V[:, keep]
    assert_cover(names)
    aux = None
    if with_aux:
        aux = {k: np.array([f.get(k, np.nan) for f in feats], np.float64)
               for k in AUX_FEATS if k in feats[0]}
    if with_aux:
        return V.astype(np.float32), names, dropped, aux
    return V.astype(np.float32), names, dropped


def cluster_index(names):
    return [cluster_of(n) for n in names]


# 归因规则（v2 修正）：按 rho^2 在**全部簇之间**成比例分摊，不做 argmax。
# argmax 会在共线量之间随机选一个（PWDB 上 kurt=0.907 与 width50=0.88 几乎并列却跨簇），
# 把整个因果权重塌到单簇，份额因此不可信。成比例分摊对共线稳健，且仍可加（和为 1）。
AMBIG_EPS = 0.05    # 次优量与最优量 |rho| 相差不足此值且跨簇时，记为归因不唯一


def cluster_weights(rho_row, names):
    """一行 rho（特征 x 物理量）-> 各簇权重，和为 1（按 rho^2 成比例）。

    用 rho^2 而不是 |rho|：rho^2 是解释方差份额，可加且不偏向符号。
    """
    pw = np.nan_to_num(rho_row, nan=0.0) ** 2
    tot = pw.sum()
    out = {}
    if tot < 1e-12:
        return out
    for c in TABLE:
        idx = [i for i, n in enumerate(names) if cluster_of(n) == c]
        if idx:
            out[c] = float(pw[idx].sum() / tot)
    return out


def top_two(rho_row, names):
    """-> (最优量, rho, 次优跨簇量, rho, 是否归因不唯一)。"""
    r = np.nan_to_num(rho_row, nan=0.0)
    if np.abs(r).max() < 1e-12:
        return "", np.nan, "", np.nan, False
    order = np.argsort(-np.abs(r))
    j = int(order[0])
    j2 = next((int(k) for k in order[1:] if cluster_of(names[k]) != cluster_of(names[j])),
              None)
    if j2 is None:
        return names[j], float(r[j]), "", np.nan, False
    amb = bool(abs(r[j]) - abs(r[j2]) < AMBIG_EPS)
    return names[j], float(r[j]), names[j2], float(r[j2]), amb


def freeze(write=True):
    sha = table_hash()
    payload = dict(version=VERSION, hash=sha, table=TABLE, concept=CONCEPT, gaps=GAPS,
                   excluded=EXCLUDED_FEATS,
                   superseded=dict(v1="3ca1c0c430f42a14", v2_0="2317524a2707afa7",
                                   v2_1="b515f83ee3b20640"),
                   change_note="v2.1->v2.2 删 4 个精确重复/镜像量（33->29）：aix_proxy、"
                               "ap_rel、p_notch_rel、t2；reflection+notch+dicrotic 合并为"
                               " reflection（8->6 簇），因为三者 PC1 在数据上不可分"
                               "（r=+1.00，参与率 1.62–3.32）。份额口径同时改为正交主成分"
                               "靶子，v2.2 的份额与 v2.0/v2.1 是不同口径，不可直接相减。")
    if write:
        with open(FREEZE_JSON, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    return payload


if __name__ == "__main__":
    import sys
    if "--freeze" in sys.argv:
        p = freeze()
        print(f"冻结完成 sha={p['hash']}  ->  {FREEZE_JSON}")
    sha = check_frozen()
    print(f"table_hash = {sha}（{'已冻结' if FROZEN_SHA else '未冻结'}）")
    nf = sum(len(v["feats"]) for v in TABLE.values())
    print(f"簇 {len(TABLE)} 个，物理量 {nf} 个")
    for c, v in TABLE.items():
        print(f"  {c:<11s} {v['family']:<7s} {v['concept']:<3s} "
              f"{len(v['feats']):>2d} 项  {v['gloss']}")
    print("中医概念映射：")
    for k, v in CONCEPT.items():
        flag = "可支撑" if v["ok"] else "缺口"
        print(f"  {k:<5s} [{flag}] <- {v['clusters']}  「{v['classic']}」")
    print("已知缺口：", "；".join(GAPS))
