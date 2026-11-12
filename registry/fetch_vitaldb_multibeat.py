"""VDB 多搏重抽：为搏间信度（ICC）下载每人多搏波形。

动机
----
vitaldb_cycles.npz 每例只存中位心动周期（1991 人中 1943 人仅 1 搏），
sdptg_b/sdptg_d 的搏间信度不可测 —— 这是规范 GAPS 明列的缺口。
本脚本从 VitalDB API 重下 ~130 例，保存每人至多 30 搏的**逐搏**周期，
供 audit_vdb_multibeat_icc.py 计算 30 量的搏间 ICC。

与 fetch_vitaldb.py 的差异（必须声明，不是抄漏）：
  1. 去掉整例 beat_consistency>=0.85 门。信度研究的对象就是搏间变异，
     用一致性门把变异大的人筛掉，ICC 会被系统性抬高 —— 那测的是
     "我们选出来的人"的信度，不是数据的信度。cons 仅记录不筛。
  2. 不再只留中位周期，逐搏保存（每人上限 30 搏，linspace 均匀取，
     避免只取段首段尾造成漂移混淆）。
  3. 保留主导心律过滤（|rr-medrr|/medrr<0.35）与 keep_frac>=0.30：
     异位搏是测量误差，不是待测的生理变异，必须剔除。

判读注意（写死，防误读）：本脚本产出的 ICC 是**搏间**信度，
与 ep9/MIMIC 的**段间**信度不是同一口径，对照表里必须分开列。
"""

import os as _os, sys as _sys

# Resolved at run time: scripts/ holds sibling groups, the repository root holds data/.
_S = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _S)
for _d in _os.listdir(_S):
    if _d[:1].isdigit() and _os.path.isdir(_os.path.join(_S, _d)):
        _sys.path.append(_os.path.join(_S, _d))
import _repo  # noqa: E402

import argparse, gzip, io, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, requests
from scipy.signal import butter, filtfilt, find_peaks

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pwfeat import resample_cycle

ROOT = _repo.ROOT
VD = os.path.join(ROOT, "data", "vitaldb")
FS = 500.0
BB, AB = butter(3, [0.5 / (FS / 2), 10.0 / (FS / 2)], btype="band")
API = "https://api.vitaldb.net/"
LO, HI = 5.0, 250.0
MAXROWS = 450_000
MAX_BEATS = 30

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0, help="只取前 limit 例（冒烟测试）")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ncases", type=int, default=130)
a = ap.parse_args()

T = pd.read_csv(os.path.join(VD, "trks.csv"))
sub = (T[T.tname == "SNUADC/ART"][["caseid", "tid"]]
       .drop_duplicates("caseid").sort_values("caseid"))
rng = np.random.default_rng(a.seed)
sel = rng.choice(len(sub), size=min(a.ncases, len(sub)), replace=False)
items = sub.iloc[np.sort(sel)].values.tolist()
if a.limit:
    items = items[: a.limit]
print(f"[multibeat] {len(items)} cases, max {MAX_BEATS} beats/case, seed={a.seed}", flush=True)

S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0"})
OUT_X = os.path.join(VD, "multibeat_cyc.npz")
OUT_F = os.path.join(VD, "multibeat_meta.csv")


def fetch(tid, tries=3):
    for k in range(tries):
        try:
            r = S.get(API + tid, timeout=180)
            if r.status_code == 200:
                raw = r.content
                return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
        except Exception:
            time.sleep(2 + 3 * k)
    return None


def clean(raw):
    df = pd.read_csv(io.BytesIO(raw), na_values=[""], dtype={0: float, 1: float},
                     nrows=MAXROWS)
    df.columns = ["t", "v"]
    v = df.v.values.copy()
    v[(v < LO) | (v > HI)] = np.nan
    s = pd.Series(v).interpolate(limit=50, limit_area="inside")
    return df.t.values, s.values


def longest_run(t, v, min_len=FS * 60):
    ok = np.isfinite(v)
    if ok.sum() < min_len:
        return None
    idx = np.where(ok)[0]
    brk = np.where(np.diff(idx) > 1)[0]
    segs, st = [], 0
    for b in brk:
        segs.append((idx[st], idx[b] + 1)); st = b + 1
    segs.append((idx[st], idx[-1] + 1))
    segs = [(i, j) for i, j in segs if j - i >= min_len]
    return max(segs, key=lambda s: s[1] - s[1] if s[1] - s[0] else 0) if segs else None


all_cyc, all_meta = [], []


def dump():
    if all_cyc:
        # X/caseid/beat 必须同为 ok 搏的行集：all_cyc 只 append ok 搏，
        # 对应的 meta 行用 ok==1 过滤后的（顺序一致）。若把失败 case 也写进
        # caseid，下游按行对齐会整体错位 —— 这正是 12/14 号 CM 版本雷的同型错误。
        ok_meta = [m for m in all_meta if m["ok"] == 1]
        np.savez_compressed(OUT_X, X=np.vstack(all_cyc).astype(np.float32),
                            caseid=np.array([m["caseid"] for m in ok_meta]),
                            beat=np.array([m["beat"] for m in ok_meta]))
    pd.DataFrame(all_meta).to_csv(OUT_F, index=False, encoding="utf-8-sig")


for n, (caseid, tid) in enumerate(items):
    raw = fetch(tid)
    if raw is None:
        all_meta.append(dict(caseid=int(caseid), beat=-1, ok=0, reason="download",
                             n_beats_kept=0, rrcv=np.nan, keep_frac=np.nan, cons=np.nan))
        continue
    try:
        t, v = clean(raw)
    except Exception as e:
        all_meta.append(dict(caseid=int(caseid), beat=-1, ok=0,
                             reason=f"parse:{type(e).__name__}", n_beats_kept=0,
                             rrcv=np.nan, keep_frac=np.nan, cons=np.nan))
        continue
    ok = np.isfinite(v)
    if ok.sum() < FS * 60:
        all_meta.append(dict(caseid=int(caseid), beat=-1, ok=0, reason="too_short",
                             n_beats_kept=0, rrcv=np.nan, keep_frac=np.nan, cons=np.nan))
        continue
    idx = np.where(ok)[0]
    brk = np.where(np.diff(idx) > 1)[0]
    segs, st = [], 0
    for b in brk:
        segs.append((idx[st], idx[b] + 1)); st = b + 1
    segs.append((idx[st], idx[-1] + 1))
    segs = [(i, j) for i, j in segs if j - i >= FS * 60]
    if not segs:
        all_meta.append(dict(caseid=int(caseid), beat=-1, ok=0, reason="too_short",
                             n_beats_kept=0, rrcv=np.nan, keep_frac=np.nan, cons=np.nan))
        continue
    i, j = max(segs, key=lambda s: s[1] - s[0])
    xr = v[i:j]
    if np.percentile(xr, 99) - np.percentile(xr, 1) < 15:
        all_meta.append(dict(caseid=int(caseid), beat=-1, ok=0, reason="flat",
                             n_beats_kept=0, rrcv=np.nan, keep_frac=np.nan, cons=np.nan))
        continue
    x = xr - np.median(xr)
    y = filtfilt(BB, AB, x)
    pk, _ = find_peaks(-y, distance=int(0.33 * FS), prominence=0.25 * np.std(y))
    if len(pk) < 6:
        all_meta.append(dict(caseid=int(caseid), beat=-1, ok=0, reason="no_beats",
                             n_beats_kept=0, rrcv=np.nan, keep_frac=np.nan, cons=np.nan))
        continue
    rr = np.diff(pk)
    rrcv = float(np.std(rr) / np.mean(rr))
    medrr = float(np.median(rr))
    pairs_all = [(p, q) for p, q in zip(pk[:-1], pk[1:]) if 0.33 * FS <= q - p <= 2.0 * FS]
    pairs = [(p, q) for p, q in pairs_all if abs((q - p) - medrr) / medrr < 0.35]
    keep_frac = len(pairs) / max(1, len(pairs_all))
    if len(pairs) < 8 or keep_frac < 0.30:
        all_meta.append(dict(caseid=int(caseid), beat=-1, ok=0,
                             reason=f"rhythm(kf={keep_frac:.2f},rrcv={rrcv:.2f})",
                             n_beats_kept=0, rrcv=rrcv, keep_frac=keep_frac, cons=np.nan))
        continue
    if len(pairs) > MAX_BEATS:
        selk = np.linspace(0, len(pairs) - 1, MAX_BEATS).astype(int)
        pairs = [pairs[k] for k in selk]
    cyc = np.vstack([resample_cycle(x[p:q]) for p, q in pairs])
    med = np.median(cyc, 0)
    cons = float(np.median([np.corrcoef(cyc[k], med)[0, 1] for k in range(len(cyc))]))
    for b in range(len(cyc)):
        all_cyc.append(cyc[b])
        all_meta.append(dict(caseid=int(caseid), beat=b, ok=1, reason="",
                             n_beats_kept=len(cyc), rrcv=rrcv,
                             keep_frac=round(keep_frac, 4),
                             cons=round(cons, 4)))
    if (n + 1) % 5 == 0:
        okc = len({m["caseid"] for m in all_meta if m["ok"] == 1})
        print(f"  {n+1}/{len(items)}  ok={okc}  beats={len(all_cyc)}", flush=True)
    if (n + 1) % 10 == 0:
        dump()

dump()
okc = len({m["caseid"] for m in all_meta if m["ok"] == 1})
print(f"[multibeat] done ok={okc}/{len(items)} beats={len(all_cyc)}", flush=True)
