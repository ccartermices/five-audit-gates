"""
真实队列周期提取（与 PWDB 共用 pwfeat.seg_feats，保证特征定义同一）
每段录音：选心脏频带 SNR 最高的通道 -> 带通 -> 逐搏分割 -> 相位归一化 -> 搏间中位数
输出：
  data/real_cycles.npz       X (n,256) 周期波形（未标定），sid/session
  data/real_features.csv     同一份 seg_feats 特征
"""

import os as _os, sys as _sys

# Resolved at run time: scripts/ holds sibling groups, the repository root holds data/.
_S = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _S)
for _d in _os.listdir(_S):
    if _d[:1].isdigit() and _os.path.isdir(_os.path.join(_S, _d)):
        _sys.path.append(_os.path.join(_S, _d))
import _repo  # noqa: E402

import numpy as np, pandas as pd, zipfile, io, os, sys, collections, warnings
warnings.filterwarnings("ignore")
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, find_peaks
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pwfeat import seg_feats, resample_cycle, NPH

ROOT = _repo.ROOT
Z = zipfile.ZipFile(os.path.join(ROOT, "25个channels被试基本信息表.zip"))
FS = 50.0
BB, AB = butter(3, [0.7 / (FS / 2), 4.0 / (FS / 2)], btype="band")

recs = []
for n in Z.namelist():
    p = n.split("/")
    if len(p) == 4 and p[3].endswith(".mat") and not p[3].startswith("Z"):
        recs.append((p[2], p[3][:-4], n))
recs.sort()
print(f"recordings = {len(recs)}")

X, rows, meta = [], [], []
for i, (sid, sess, name) in enumerate(recs):
    m = loadmat(io.BytesIO(Z.read(name)))
    k = [x for x in m if not x.startswith("__")][0]
    A = np.asarray(m[k], float)
    if A.ndim != 2 or A.shape[0] < 200:
        continue
    # 选心脏频带 SNR 最高的通道
    snr = np.empty(A.shape[1])
    for c in range(A.shape[1]):
        y = filtfilt(BB, AB, A[:, c] - A[:, c].mean())
        snr[c] = np.var(y) / (np.var(A[:, c] - A[:, c].mean()) + 1e-12)
    c = int(np.argmax(snr))
    y = filtfilt(BB, AB, A[:, c] - A[:, c].mean())
    # 足点（舒张末极小）对齐，与 PWDB 的 onset 定义一致
    pk, _ = find_peaks(-y, distance=int(0.4 * FS), prominence=0.35 * np.std(y))
    if len(pk) < 4:
        continue
    cyc = []
    for a, b in zip(pk[:-1], pk[1:]):
        seg = A[a:b, c]
        if len(seg) < int(0.3 * FS) or len(seg) > int(1.5 * FS):
            continue
        cyc.append(resample_cycle(seg))
    if len(cyc) < 3:
        continue
    cyc = np.vstack(cyc)
    med = np.median(cyc, axis=0)
    rr = np.diff(pk) / FS
    f = seg_feats(med)
    f["cyc_dur"] = float(np.median(rr))
    f["hr"] = 60.0 / f["cyc_dur"]
    f["n_beats"] = len(cyc)
    f["rr_cv"] = float(np.std(rr) / (np.mean(rr) + 1e-12))
    f["beat_consistency"] = float(np.median(
        [np.corrcoef(cyc[j], med)[0, 1] for j in range(len(cyc))]))
    f["chosen_ch"] = c
    f["snr_best"] = float(snr[c])
    f["sid"], f["session"] = sid, sess
    rows.append(f)
    X.append(med)
    meta.append((sid, sess))
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(recs)}")

R = pd.DataFrame(rows)
print("valid recordings", len(R), "/", len(recs))
print("beats/rec median", int(R.n_beats.median()), "| beat_consistency median",
      round(float(R.beat_consistency.median()), 3),
      "| rr_cv median", round(float(R.rr_cv.median()), 3))
R.to_csv(os.path.join(ROOT, "data", "real_features.csv"), index=False, encoding="utf-8-sig")
np.savez_compressed(os.path.join(ROOT, "data", "real_cycles.npz"),
                    X=np.vstack(X).astype(np.float32),
                    sid=np.array([m[0] for m in meta]),
                    session=np.array([m[1] for m in meta]))
print("saved real_features.csv / real_cycles.npz", np.vstack(X).shape)
