"""
共享脉搏波特征模块 —— 两个域必须调用同一份实现，否则跨域比较无效。
"""

import os as _os, sys as _sys

# Resolved at run time: scripts/ holds sibling groups, the repository root holds data/.
_S = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _S)
for _d in _os.listdir(_S):
    if _d[:1].isdigit() and _os.path.isdir(_os.path.join(_S, _d)):
        _sys.path.append(_os.path.join(_S, _d))
import _repo  # noqa: E402

import numpy as np

NPH = 256


def resample_cycle(a, n=NPH):
    a = np.asarray(a, float).ravel()
    if len(a) == n:
        return a
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a)


def p2_index(p):
    """定位收缩峰 P1、反射波峰 P2、切迹，返回 (i1, i2, jn)。

    单独提出来，是因为 P2/切迹的**定位**与**测量**必须能被分别调用：
    定位对高频抖动极敏感（量化台阶会造出伪局部极大），测量则希望少滤波。
    消融实验需要同一份定位逻辑，不能在别处再抄一遍。
    """
    p = np.asarray(p, float)
    n = len(p)
    pmin, pmax = p.min(), p.max()
    q = (p - pmin) / (pmax - pmin + 1e-12)
    i1 = int(np.argmax(q[: int(0.60 * n)]))          # 收缩峰 P1
    # P2 = 收缩期反射波肩部（切迹之前的局部极大中最高者），这是 AIx 的标准定义
    # 注意：量化数据在峰顶有并列平台（多个采样点 q=1.0），必须先跳过平台再找局部极大
    e = min(n - max(2, int(0.10 * n)), i1 + int(0.75 * n))
    plat = 0.02 * (q.max() - q.min() + 1e-12) + 0.02
    k0 = i1 + 1
    while k0 < e and q[k0] > q[i1] - plat:
        k0 += 1
    cand = [i for i in range(k0 + 1, e - 1)
            if q[i] > q[i - 1] - 1e-12 and q[i] >= q[i + 1] - 1e-12]
    i2 = cand[int(np.argmax([q[i] for i in cand]))] if cand else i1
    if i2 > i1 + 1:
        jn = i1 + 1 + int(np.argmin(q[i1 + 1: i2]))
    else:
        jn = i1
    return i1, i2, jn, q


def sdptg_waves(p, k=9):
    """单周期 -> SDPTG（二阶导）五波幅度，按 a 归一。

    v2.3 起是 b/a 与 d/a 的唯一实现（19 号 §6；原在 audit_v22_sdptg.py）。
    收缩期找主正峰 a，其后按符号交替找 b(负) c(正) d(负)，舒张期找 e(正)。
    找不到返回 nan，不补零 —— 补零会伪造"重搏平坦"。
    k 为段内平滑窗宽（采样点）；二阶导对量化台阶极敏感，k=9 是
    audit_v22_sdptg 三档 {5,9,13} 的中档。
    """
    p = np.asarray(p, float)
    pm, px = p.min(), p.max()
    q = (p - pm) / (px - pm + 1e-12)
    n = len(q)
    if k > 1:
        qs = np.convolve(q, np.ones(k) / k, mode="same")
    else:
        qs = q
    d2 = np.gradient(np.gradient(qs))
    i1 = int(np.argmax(d2[: int(0.55 * n)]))
    a = d2[i1]
    if a <= 0:
        return dict(a=np.nan, b=np.nan, c=np.nan, d=np.nan, e=np.nan)

    def first_signed(start, sign, min_frac=0.04):
        m = max(2, int(min_frac * n))
        for i in range(start + 1, n - 2):
            if sign > 0 and d2[i] >= d2[i - 1] and d2[i] > d2[i + 1] and d2[i] > 0:
                return i, d2[i]
            if sign < 0 and d2[i] <= d2[i - 1] and d2[i] < d2[i + 1] and d2[i] < 0:
                return i, d2[i]
        return -1, np.nan

    ib, b = first_signed(i1 + 2, -1)
    ic, c = first_signed(ib + 2, +1) if ib > 0 else (-1, np.nan)
    id_, d = first_signed(ic + 2, -1) if ic > 0 else (-1, np.nan)
    ie, e = first_signed(id_ + 2, +1) if id_ > 0 else (-1, np.nan)
    return dict(a=float(a), b=float(b / a), c=float(c / a),
                d=float(d / a), e=float(e / a))


def seg_feats(p):
    """p: 单周期压波形（任意单位）。返回全部无量纲特征 + 绝对幅值两项。"""
    p = np.asarray(p, float)
    n = len(p)
    pmin, pmax = p.min(), p.max()
    pp = pmax - pmin
    q = (p - pmin) / pp
    t = np.arange(n) / (n - 1)
    f = {}
    f["pp_mmHg"] = pp
    f["mbp_mmHg"] = p.mean()
    q0 = q - q.mean()
    Fh = np.abs(np.fft.rfft(q0)) / (n / 2)
    for k in range(1, 9):
        f[f"h{k}"] = Fh[k] if k < len(Fh) else 0.0
    h1 = max(f["h1"], 1e-12)
    for k in range(2, 9):
        f[f"h{k}_r"] = f[f"h{k}"] / h1
    f["hf_energy"] = np.sqrt(sum(f[f"h{k}"] ** 2 for k in range(2, 9))) / h1
    i1, i2, jn, q = p2_index(p)
    f["t1"] = t[i1]
    f["t2"] = t[i2]
    # P2 未检出：i2 = i1（cand 为空，i1 之后没有局部极大）。实测 CM 有 23.5% 的段
    # 落在这里。此前把这几个量照算，p2_rel 就恒为 1.0 —— 那不是"AIx=100%"，是
    # "测不出"。伪造出的 1.0 会一边拉低归因 rho、一边在人群排序里冒充真实观测，
    # 所以三个增强指数在此置缺失，并用 p2_found 显式记录，交由下游决定怎么用。
    # 注意：notch_depth 不置缺失 —— "没有 P2"与"没有切迹"在形态上是同一件事，
    # 归零是它的正确取值，不是伪值。
    found = bool(i2 > i1)
    f["p2_found"] = int(found)
    if found:
        # AIx 标准式 (P2-DBP)/(P1-DBP)。分母用 q[i1] 而非 1：当前 P1 恒为全局峰
        # （q1≡1，见消融表4），两种写法数值相同；写成比值是为防未来 P1 定位改动后
        # 悄悄引入上溢，代价为零。
        f["p2_rel"] = float(q[i2] / max(q[i1], 1e-9))
        f["ap_rel"] = float(q[i2] - q[i1])
        f["aix_proxy"] = float(q[i2] - q[i1])
    else:
        f["p2_rel"] = float("nan")
        f["ap_rel"] = float("nan")
        f["aix_proxy"] = float("nan")
    f["t_notch"] = t[jn]
    f["p_notch_rel"] = float(q[jn])
    f["notch_depth"] = float(q[i1] - q[jn])
    # 切迹之后的舒张期峰（重搏波），单独记，不参与 AIx
    e2 = n - max(2, int(0.05 * n))
    i3 = jn + int(np.argmax(q[jn: e2])) if e2 > jn + 2 else jn
    f["t_dicrotic"] = t[i3]
    f["p_dicrotic_rel"] = float(q[i3])
    rise = q[: i1 + 1]
    f["upstroke_slope"] = (rise.max() - rise.min()) * (n - 1) / max(1, i1)
    f["t_90"] = t[int(np.argmax(q >= 0.9))] if (q >= 0.9).any() else np.nan
    f["t_50up"] = t[int(np.argmax(q >= 0.5))] if (q >= 0.5).any() else np.nan
    dec = q[i1:]
    f["t_50down"] = t[i1 + int(np.argmax(dec <= 0.5))] if (dec <= 0.5).any() else np.nan
    f["width50"] = (f["t_50down"] - f["t_50up"]) if not (
        np.isnan(f["t_50down"]) or np.isnan(f["t_50up"])) else np.nan
    qm, qs = q.mean(), q.std() + 1e-12
    f["skew"] = float(((q - qm) ** 3).mean() / qs ** 3)
    f["kurt"] = float(((q - qm) ** 4).mean() / qs ** 4)
    cnt = np.histogram(q, 20, (0, 1), density=True)[0] + 1e-12
    f["q_entropy"] = float(-(cnt * np.log(cnt)).sum())
    # v2.3 新增：SDPTG 二阶导指数。b/a = 早期反射，d/a = 晚期(重搏)波。
    # 不依赖显式切迹 —— 这是它们相对 p_dicrotic_rel/t_dicrotic 的存在理由。
    w = sdptg_waves(p)
    f["sdptg_b"] = w["b"]
    f["sdptg_d"] = w["d"]
    return f
