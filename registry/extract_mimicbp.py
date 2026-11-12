"""
MIMIC-BP 抽取：把 30s@125Hz 的 ABP 段切成逐搏周期，用与其余三源完全相同的
pwfeat.seg_feats 提特征，保证"同一实现、同一足点约定、同一相位归一化"。

输入  data/mimicbp/abp/<pid>_abp.npy      (30, 3750) @125Hz
      data/mimicbp/labels/<pid>_labels.npy (30, 2)   [SBP, DBP]
输出  data/mimicbp_cycles.npz   X (n,256) float32, pid, seg
      data/mimicbp_features.csv 同 schema 特征 + 质量字段 + 逐段 SBP/DBP
"""

import os as _os, sys as _sys

# Resolved at run time: scripts/ holds sibling groups, the repository root holds data/.
_S = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _S)
for _d in _os.listdir(_S):
    if _d[:1].isdigit() and _os.path.isdir(_os.path.join(_S, _d)):
        _sys.path.append(_os.path.join(_S, _d))
import _repo  # noqa: E402

import os, sys, glob, csv, warnings
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pwfeat import seg_feats, resample_cycle, NPH

ROOT = _repo.ROOT
BASE = os.path.join(ROOT, "data", "mimicbp")
FS = 125.0
BB, AB = butter(3, [0.7 / (FS / 2), 4.0 / (FS / 2)], btype="band")
SEG_LEN = 3750
MIN_BEATS = 8


def find_abp_dir():
    for cand in ["abp/abp", "abp", "."]:
        p = os.path.join(BASE, cand)
        if glob.glob(os.path.join(p, "*_abp.npy")):
            return p
    hits = glob.glob(os.path.join(BASE, "**", "*_abp.npy"), recursive=True)
    return os.path.dirname(hits[0]) if hits else None


def find_label_dir():
    p = os.path.join(BASE, "labels")
    if glob.glob(os.path.join(p, "*_labels.npy")):
        return p
    hits = glob.glob(os.path.join(BASE, "**", "*_labels.npy"), recursive=True)
    return os.path.dirname(hits[0]) if hits else None


ABP_DIR, LBL_DIR = find_abp_dir(), find_label_dir()
print("ABP_DIR =", ABP_DIR)
print("LBL_DIR =", LBL_DIR)
assert ABP_DIR and LBL_DIR, "未找到 MIMIC-BP 解压目录，请先解压 abp.zip / labels.zip"

files = sorted(glob.glob(os.path.join(ABP_DIR, "*_abp.npy")))
print("subjects =", len(files))

X, rows, meta = [], [], []
n_seg_ok = n_seg_bad = 0

for i, fp in enumerate(files):
    pid = os.path.basename(fp).replace("_abp.npy", "")
    A = np.load(fp)
    lf = os.path.join(LBL_DIR, pid + "_labels.npy")
    if not os.path.exists(lf):
        continue
    L = np.load(lf)
    if A.ndim == 1:
        A = A[None, :]
    if A.shape[1] != SEG_LEN:
        continue
    for s in range(min(len(A), len(L))):
        y0 = np.asarray(A[s], float)
        # 有创 ABP：无标定的段（全 0 / 恒值 / 越界）直接跳过
        if not np.isfinite(y0).all():
            m = np.isfinite(y0)
            if m.sum() < 0.8 * SEG_LEN:
                n_seg_bad += 1
                continue
            y0 = np.interp(np.arange(SEG_LEN), np.flatnonzero(m), y0[m])
        if np.percentile(y0, 99) < 20 or np.percentile(y0, 1) > 300:
            n_seg_bad += 1
            continue
        if np.ptp(y0) < 10:
            n_seg_bad += 1
            continue
        y = filtfilt(BB, AB, y0 - y0.mean())
        pk, _ = find_peaks(-y, distance=int(0.4 * FS), prominence=0.35 * np.std(y))
        if len(pk) < MIN_BEATS + 1:
            n_seg_bad += 1
            continue
        cyc = []
        for a, b in zip(pk[:-1], pk[1:]):
            seg = y0[a:b]
            if len(seg) < int(0.30 * FS) or len(seg) > int(1.50 * FS):
                continue
            cyc.append(resample_cycle(seg))
        if len(cyc) < MIN_BEATS:
            n_seg_bad += 1
            continue
        cyc = np.vstack(cyc)
        med = np.median(cyc, axis=0)
        rr = np.diff(pk) / FS
        f = seg_feats(med)
        f["cyc_dur"] = float(np.median(rr))
        f["hr"] = 60.0 / f["cyc_dur"]
        f["n_beats"] = int(len(cyc))
        f["rr_cv"] = float(np.std(rr) / (np.mean(rr) + 1e-12))
        f["keep_frac"] = 1.0
        f["beat_consistency"] = float(np.median(
            [np.corrcoef(cyc[j], med)[0, 1] for j in range(len(cyc))]))
        f["valid_frac"] = 1.0
        f["dur_s"] = SEG_LEN / FS
        f["sbp"] = float(L[s, 0])
        f["dbp"] = float(L[s, 1])
        f["pid"], f["seg"] = pid, s
        rows.append(f)
        X.append(med)
        meta.append((pid, s))
        n_seg_ok += 1
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(files)} subjects, segments kept={n_seg_ok}")

print(f"segments kept={n_seg_ok} dropped={n_seg_bad}")

FEATS = None
with open(os.path.join(ROOT, "data", "mimicbp_features.csv"), "w", newline="",
          encoding="utf-8-sig") as fh:
    cols = list(rows[0].keys())
    FEATS = cols
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

np.savez_compressed(
    os.path.join(ROOT, "data", "mimicbp_cycles.npz"),
    X=np.vstack(X).astype(np.float32),
    pid=np.array([m[0] for m in meta]),
    seg=np.array([m[1] for m in meta]),
)
print("saved mimicbp_features.csv", len(rows), "rows;",
      "mimicbp_cycles.npz", np.vstack(X).shape)
