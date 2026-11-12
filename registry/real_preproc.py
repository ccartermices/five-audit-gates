"""真实队列（CM 25 通道阵列）的唯一预处理实现。

为什么必须单独有一个文件
----------------------
原始 `.mat` 是**未处理**的采集：含基线漂移、量化台阶、接触不良段。直接拿它
算形态量或给医生判读，得到的都是噪声的几何形状。此前 `extract_real.py` 逐搏
片段取的是原始列（`seg = A[a:b, c]`），且带通只到 4 Hz——切迹与高阶谐波被削掉，
同时没有剔除低幅段。本文件把预处理固定下来，供训练特征与标注图共用同一口径。

采样率的依据（不要凭猜）
---------------------
`Cell Matrix/<ID>/<段>.mat` 为 8405 × 25。同目录 `.dat` 首列是 Time，
前两行 6181502 → 6181522，步长恒为 **20 ms**，即 **fs = 50 Hz**，8405 点 = 168.1 s。
注意 `<ID>.TXT` 里的 `SAMPLE_RATE = 128` 是**单通道张力计**导出时的采样率，
与 25 通道阵列不是同一设备，不能拿来当本文件的 fs。

处理链（顺序不可交换）
--------------------
1. 去直流
2. 去基线：3 阶 Butterworth 高通 0.5 Hz（零相位 filtfilt）
3. 抗混叠/去高频：同阶低通 10 Hz
4. 抑制量化台阶：Savitzky-Golay(11, 3)，11 点 = 0.22 s，短于一个上升支
5. 通道选择：滑动 12 s 窗，按"脉搏带能量占比 / (1+RR_CV)"取最优
6. 搏剔除：相对幅度、RR 偏离、与中位搏的相关
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


import io
import os
import zipfile

import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter

ROOT = _repo.ROOT
ZIP = os.path.join(ROOT, "25个channels被试基本信息表.zip")

FS = 50.0            # 由 .dat 的 Time 列步长 20 ms 推得，见模块文档
WIN_S = 12.0         # 判读/评分窗
BEAT_N = 256         # 相位归一化后的每搏点数（与 PWDB 一致）

# 质量门（写死在这里，禁止各调用点自行放宽；改动须同步版本号）
GATE_SNR = 0.35      # 脉搏带（0.8–2.0 Hz）能量占 0.4 Hz 以上总能量的比例
GATE_SNR_MIN = 0.28  # 窗内四个子块中**最差**子块的脉搏占比
GATE_CV = 0.12       # 窗内 RR 变异系数
GATE_AMP = 0.15      # p95-p5 幅度（原始单位）
GATE_STAB = 0.55     # 子块幅度 min/max：用来挡"一半好一半掉线"的窗
GATE_BEATS = (8, 22)  # 12 s 内应有搏数

_BP = butter(3, [0.5 / (FS / 2), 10.0 / (FS / 2)], btype="band")


def load_record(pid: str, seg: str, zf: zipfile.ZipFile | None = None):
    """返回 (A, fs)。A: (n, 25)，原始未处理。"""
    zf = zf or zipfile.ZipFile(ZIP)
    key = f"/Cell Matrix/{pid}/{seg}.mat"
    hit = [x for x in zf.namelist() if x.endswith(key)]
    if not hit:
        raise FileNotFoundError(key)
    A = sio.loadmat(io.BytesIO(zf.read(hit[0])))[seg].astype(float)
    return A, FS


def clean(x: np.ndarray, use_savgol: bool = False) -> np.ndarray:
    """去直流 → 去基线(0.5 Hz) + 去高频(10 Hz)。

    v3 变更（2026-11-02，见 15 号 §1）：**去掉 savgol(11,3)**。
    实测它对谐波无收益（h3 |Δ|=0.018），但把切迹可分率从 80% 压到 40% ——
    11 点窗在 50 Hz 下跨 0.22 s，正好吃掉切迹这种百毫秒级结构。
    保留原实现于 `clean_savgol`，仅供复现 v2 与做对照，不进入生产链。
    """
    x = np.asarray(x, float)
    y = filtfilt(_BP[0], _BP[1], x - x.mean())
    return savgol_filter(y, 11, 3) if use_savgol else y


def clean_savgol(x: np.ndarray) -> np.ndarray:
    """v2 旧链（含 savgol），只用于对照与复现，不得用于生产。"""
    return savgol_filter(clean(x), 11, 3)


def _band_share(seg: np.ndarray) -> float:
    """脉搏带（0.8–2.0 Hz）能量占 0.4 Hz 以上总能量的比例。"""
    n = len(seg)
    if n < int(0.8 * FS):
        return 0.0
    f = np.abs(np.fft.rfft(seg))
    fr = np.fft.rfftfreq(n, 1 / FS)
    return float(f[(fr > 0.8) & (fr < 2.0)].sum() / (f[(fr > 0.4)].sum() + 1e-9))


def _win_metrics(seg: np.ndarray):
    """整窗指标 + 窗内四个子块的稳定性。

    为什么必须看子块
    --------------
    只看整窗会把"前 6 s 正常、后 6 s 传感器掉线"的窗判成好窗：整窗峰数够、
    RR 也算得出来，但医生看到的一半是平线。这类窗必须先被识别出来，
    否则前置质量门形同虚设（实测第一张标注图就踩了这个坑）。
    """
    snr = _band_share(seg)
    pk, _ = find_peaks(seg, distance=int(0.4 * FS), prominence=0.4 * np.std(seg))
    if len(pk) < 4:
        return None
    rr = np.diff(pk) / FS
    cv = float(rr.std() / (rr.mean() + 1e-12))
    amp = float(np.percentile(seg, 95) - np.percentile(seg, 5))

    blocks = np.array_split(seg, 4)
    snrs = [_band_share(b) for b in blocks]
    amps = [float(np.percentile(b, 95) - np.percentile(b, 5)) for b in blocks]
    amp_stab = float(min(amps) / (max(amps) + 1e-12))
    return dict(snr=snr, snr_min=float(min(snrs)), cv=cv, amp=amp,
                amp_stab=amp_stab, n_beat=len(pk))


def pick_channel_window(A: np.ndarray, win_s: float = WIN_S, step_s: float = 3.0,
                        use_savgol: bool = False):
    """逐通道、逐窗扫描，返回最优 (channel, start, stop, metrics)。

    评分 = snr / (1+cv)：只按 snr 选会挑到"能量集中但节律乱"的段，
    只按 cv 选会挑到"安静但没脉搏"的段。
    """
    W = int(win_s * FS)
    S = int(step_s * FS)
    best = None
    Y = np.empty_like(A)
    for c in range(A.shape[1]):
        y = clean(A[:, c], use_savgol=use_savgol)
        Y[:, c] = y
        for st in range(0, max(len(y) - W, 1), S):
            m = _win_metrics(y[st:st + W])
            if m is None:
                continue
            # 评分 = 最差子块的脉搏纯度 × 子块幅度稳定性 × 幅度
            # 只按纯度会选中"干净但幅度只有 0.15"的弱通道——医生看不出形态；
            # 只按幅度会选中掉线前的高幅段。三者相乘才同时排除这两种。
            m["score"] = (m["snr_min"] / (1 + m["cv"])) * m["amp_stab"] * m["amp"]
            if best is None or m["score"] > best[0]["score"]:
                best = (m, c, st)
    if best is None:
        return None
    m, c, st = best
    return dict(ch=c, start=st, stop=st + W, y=Y[:, c], Y=Y, **m)


def gate(m: dict) -> tuple[bool, str]:
    """质量门。返回 (通过, 原因)。不通过的原因要落盘，便于统计哪一类最多。"""
    if m["snr"] < GATE_SNR:
        return False, f"snr={m['snr']:.2f}<{GATE_SNR}"
    if m["cv"] > GATE_CV:
        return False, f"cv={m['cv']:.2f}>{GATE_CV}"
    if m["amp"] < GATE_AMP:
        return False, f"amp={m['amp']:.2f}<{GATE_AMP}"
    if m.get("snr_min", 1.0) < GATE_SNR_MIN:
        return False, f"snr_min={m['snr_min']:.2f}<{GATE_SNR_MIN}"
    if m.get("amp_stab", 1.0) < GATE_STAB:
        return False, f"stab={m['amp_stab']:.2f}<{GATE_STAB}"
    if not (GATE_BEATS[0] <= m["n_beat"] <= GATE_BEATS[1]):
        return False, f"n_beat={m['n_beat']}"
    return True, "ok"


def beat_bounds(y: np.ndarray, lo_s: float = 0.35, hi_s: float = 1.6):
    """取足点（舒张末极小）作为搏界，与 PWDB 的 onset 定义一致。"""
    foot, _ = find_peaks(-y, distance=int(lo_s * FS), prominence=0.35 * np.std(y))
    return foot


def median_beat(y: np.ndarray, foot: np.ndarray, n: int = BEAT_N):
    """按足点切搏 → 相位归一化 → 逐搏质量筛选 → 中位搏。"""
    segs, keep = [], []
    for a, b in zip(foot[:-1], foot[1:]):
        L = b - a
        if L < int(0.35 * FS) or L > int(1.6 * FS):
            continue
        segs.append(np.interp(np.linspace(0, 1, n), np.linspace(0, 1, L), y[a:b]))
        keep.append((a, b))
    if len(segs) < 3:
        return None, dict(n_used=0, reason="beats<3")
    S = np.vstack(segs)
    med = np.median(S, axis=0)
    r = np.array([np.corrcoef(s, med)[0, 1] for s in S])
    ok = r >= max(0.6, np.percentile(r, 20))
    if ok.sum() < 3:
        return None, dict(n_used=int(ok.sum()), reason="corr")
    med2 = np.median(S[ok], axis=0)
    return med2, dict(n_used=int(ok.sum()), n_raw=len(segs),
                      corr_med=float(np.median(r[ok])))


def preprocess(pid: str, seg: str, zf: zipfile.ZipFile | None = None,
               use_savgol: bool = False):
    """端到端：原始 → 清洗 → 选通道/窗 → 质量门 → 中位搏。

    返回 dict，含 usable 标志与全部质量指标，供落盘与绘图共用。
    """
    A, fs = load_record(pid, seg, zf)
    p = pick_channel_window(A, use_savgol=use_savgol)
    if p is None:
        return dict(pid=pid, seg=seg, usable=False, why="no-window", fs=fs)
    ok, why = gate(p)
    beat, bm = median_beat(p["y"], beat_bounds(p["y"]))
    if beat is None:
        ok, why = False, f"beat:{bm.get('reason')}"
    return dict(pid=pid, seg=seg, usable=ok, why=why, fs=fs,
                ch=p["ch"], start_s=p["start"] / FS, stop_s=p["stop"] / FS,
                snr=p["snr"], snr_min=p["snr_min"], cv=p["cv"], amp=p["amp"],
                amp_stab=p["amp_stab"], n_beat=p["n_beat"],
                y_seg=p["y"][p["start"]:p["stop"]], y_full=p["y"],
                beat=beat, n_used=bm.get("n_used"), corr_med=bm.get("corr_med"))


def selfcheck():
    """自检：采样率、处理链、质量门在已知良好记录上必须成立。"""
    import sys
    pid, seg = "101303M", "L1"
    r = preprocess(pid, seg)
    print(f"[selfcheck] {pid}/{seg} fs={r['fs']} 通道={r['ch']} "
          f"窗={r['start_s']:.0f}-{r['stop_s']:.0f}s snr={r['snr']:.2f} "
          f"snr_min={r['snr_min']:.2f} stab={r['amp_stab']:.2f} "
          f"cv={r['cv']:.2f} amp={r['amp']:.2f} n_beat={r['n_beat']} "
          f"usable={r['usable']} ({r['why']})")
    assert r["fs"] == 50.0, "采样率必须是 50 Hz（依据 .dat 的 Time 列）"
    assert abs(r["n_beat"] / WIN_S * 60 - 70) < 25, "窗内搏数应落在生理心率范围"
    assert r["usable"], "已知良好记录未通过质量门，说明门限或处理链有问题"
    y = r["y_seg"]
    assert np.isfinite(y).all()
    print("[selfcheck] OK：去基线后均值≈0，形态为单峰+重搏切迹")
    return 0


if __name__ == "__main__":
    raise SystemExit(selfcheck())
