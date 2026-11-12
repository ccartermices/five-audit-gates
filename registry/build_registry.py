"""论文数字登记表（registry）的构建器。

为什么要有这个
--------------
23 号暴露了一类反复出现的风险：文档里的数字取自「某一次运行的屏幕输出」，
事后既复现不了、也无从判断它是从哪个文件算出来的。中医这篇要投稿，
必须做到「每个数字都能被一条命令重新算出来，且算不出来就报错」。

设计
----
`registry/numbers.yaml` 只写**取数表达式**，不写数值。每个条目：

    - key: cm_rho_h1
      claim: "CM 人级 h1→SBP 的秩相关"
      doc: "19"                 # 出自哪份编号文档
      level: main               # main / secondary / descriptive
      unit: ""
      fmt: float3
      op: pyexpr
      expr: float(csv_("audit_cm_person_level.csv").set_index("model").loc["h1only","rho"])
      bounds: [-1, 1]

表达式可用的东西（仅此几项，无 __builtins__）：
    csv_(name)  读 data/ 下的 CSV（utf-8-sig，自动去 BOM）
    npz_(name)  读 data/ 下的 npz
    json_(name) 读 data/ 下的 json
    PL          scripts/phys_layer.py（冻结层，哈希可核）
    np, pd      numpy / pandas
    K           已完成条目的取值（按 yaml 顺序，可用于派生量）

构建产物
--------
registry/numbers.json   每个 key 的取值 + 文本 + 来源文件及其哈希
registry/numbers.md     人读表
registry/numbers.tex    LaTeX 宏（键名转成合法控制词）
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
import io
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

ROOT = _repo.ROOT
DATA = os.environ.get("REGISTRY_DATA") or os.path.join(ROOT, "data")
REG = os.path.join(ROOT, "registry")
SPEC = os.path.join(REG, "numbers.yaml")

sys.path.insert(0, os.path.join(ROOT, "scripts"))
import phys_layer as PL  # noqa: E402

FMT = {
    "int": lambda v: f"{int(round(float(v)))}",
    "float1": lambda v: f"{float(v):.1f}",
    "float2": lambda v: f"{float(v):.2f}",
    "float3": lambda v: f"{float(v):.3f}",
    "float4": lambda v: f"{float(v):.4f}",
    "pct1": lambda v: f"{100 * float(v):.1f}%",
    "pct2": lambda v: f"{100 * float(v):.2f}%",
    "text": str,
}


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


class Sources:
    """一次求值过程中被读到的文件，用于 provenance 与过期检测。"""

    def __init__(self):
        self.paths = []

    def use(self, path):
        p = os.path.abspath(path)
        if p not in self.paths:
            self.paths.append(p)
        return p


def make_env(sources, known):
    cache_csv, cache_npz, cache_json = {}, {}, {}

    def _path(name):
        p = os.path.join(DATA, name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"registry 引用的数据文件不存在：{p}")
        sources.use(p)
        return p

    def csv_(name):
        if name not in cache_csv:
            cache_csv[name] = pd.read_csv(_path(name), encoding="utf-8-sig")
        return cache_csv[name].copy()

    def npz_(name):
        if name not in cache_npz:
            cache_npz[name] = np.load(_path(name), allow_pickle=True)
        return cache_npz[name]

    def json_(name):
        if name not in cache_json:
            with open(_path(name), encoding="utf-8") as f:
                cache_json[name] = json.load(f)
        return cache_json[name]

    # 只放行纯计算用的内建函数；没有 open/exec/eval/import，
    # 表达式再怎么写也只能读 csv_/npz_/json_/PL 这几条受控通道。
    SAFE = {name: __builtins__[name] if isinstance(__builtins__, dict)
            else getattr(__builtins__, name)
            for name in ("float", "int", "str", "len", "sum", "min", "max",
                         "round", "abs", "sorted", "list", "set", "dict",
                         "range", "enumerate", "zip", "bool", "tuple")}
    env = {"csv_": csv_, "npz_": npz_, "json_": json_,
           "np": np, "pd": pd, "PL": PL, "K": known,
           "__builtins__": SAFE}
    return env


def evaluate(entry, known):
    """求值单个条目，返回 (value, sources)。literal 条目无来源文件。"""
    op = entry.get("op", "pyexpr")
    src = Sources()
    if op == "literal":
        # 只允许用于「无法从数据文件复取」的量，且必须写 locator；
        # check_registry 会对 main 级 literal 直接判 BLOCK。
        return entry["value"], src.paths
    if op == "const":
        # 从源码里取模块级常量。比 import 更好：probe_mediator 会拖 torch，
        # 而常量就是一行 `NAME = 2.6`，正则取出来 + 记文件哈希即可追溯。
        import re
        p = os.path.join(ROOT, entry["file"])
        src.use(p)
        txt = io.open(p, encoding="utf-8").read()
        m = re.search(rf"^{re.escape(entry['name'])}\s*=\s*([-\d.eE+]+)", txt, re.M)
        if not m:
            raise ValueError(f"在 {entry['file']} 里找不到常量 {entry['name']}")
        return float(m.group(1)), src.paths
    if op != "pyexpr":
        raise ValueError(f"{entry['key']}: 未知 op {op!r}")
    # 表达式若用到冻结层，就把 phys_layer.py 记为来源——否则「改了特征定义」
    # 这种事不会触发 C5 过期检测。
    if "PL." in (entry.get("expr") or ""):
        src.use(os.path.join(ROOT, "scripts", "phys_layer.py"))
    for dep in entry.get("deps", []) or []:
        src.use(os.path.join(ROOT, dep))
    env = make_env(src, known)
    val = eval(entry["expr"], env)  # noqa: S307 — 表达式来自本仓库的登记表
    if isinstance(val, (np.generic,)):
        val = val.item()
    if isinstance(val, (np.ndarray, pd.Series)):
        val = float(np.asarray(val).ravel()[0])
    return val, src.paths


def render(value, fmt):
    if fmt.startswith("expr:"):
        return fmt[5:].format(v=value)
    return FMT[fmt](value)


def _rel(p):
    """相对路径；跨盘符（临时目录在 C:，仓库在 E:）时退回绝对路径。"""
    try:
        return os.path.relpath(p, ROOT)
    except ValueError:
        return os.path.abspath(p)


def build(spec_path=SPEC, out_dir=REG, write=True, built_at=None):
    with open(spec_path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    entries = spec["entries"]
    values, known, out = {}, {}, {}
    src_map = {}
    problems = []
    deferred = []  # op: registry —— 值来自「本轮登记表自身」，必须等全部条目算完

    for e in entries:
        key = e["key"]
        if key in out:
            problems.append(f"重复 key：{key}")
            continue
        if e.get("op") == "registry":
            deferred.append(e)
            continue
        try:
            val, srcs = evaluate(e, known)
        except Exception as ex:  # noqa: BLE001
            problems.append(f"{key}: 求值失败 {type(ex).__name__}: {ex}")
            continue
        if isinstance(val, float) and not np.isfinite(val):
            problems.append(f"{key}: 得到非有限值 {val}")
            continue
        txt = render(val, e.get("fmt", "float3"))
        # 派生量继承上游条目的来源：上游数据一改，派生量同样失效
        for up in e.get("deps_from", []) or []:
            for p in src_map.get(up, []):
                if p not in srcs:
                    srcs.append(p)
        values[key] = val
        known[key] = val
        src_map[key] = list(srcs)
        out[key] = dict(
            key=key, value=val, text=txt,
            claim=e.get("claim", ""), doc=e.get("doc", ""),
            level=e.get("level", "secondary"),
            unit=e.get("unit", ""), fmt=e.get("fmt", "float3"),
            section=e.get("section", ""),
            op=e.get("op", "pyexpr"),
            expr=e.get("expr", None),
            locator=e.get("locator", None),
            note=e.get("note", ""),
            sources=[dict(path=_rel(p), sha16=sha16(p)) for p in srcs],
        )

    # ---- 自指条目：条目数、literal 数、来源文件数 ----
    # 这些量只有在本轮别的条目都算完之后才存在，所以单独一批处理；
    # 来源记为 numbers.yaml 本身，yaml 一改 C5 立刻报过期。
    facts = dict(n_entries=len(out) + len(deferred),
                 n_main=(sum(1 for v in out.values() if v["level"] == "main")
                         + sum(1 for e in deferred if e.get("level") == "main")),
                 n_literal=sum(1 for v in out.values() if v["op"] == "literal"),
                 n_sources=len({p for ps in src_map.values() for p in ps})
                 + (1 if deferred else 0))
    for e in deferred:
        key, name = e["key"], e.get("name")
        if name not in facts:
            problems.append(f"{key}: 未知 name {name!r}")
            continue
        val = facts[name]
        out[key] = dict(
            key=key, value=val, text=render(val, e.get("fmt", "int")),
            claim=e.get("claim", ""), doc=e.get("doc", ""),
            level=e.get("level", "secondary"),
            unit=e.get("unit", ""), fmt=e.get("fmt", "int"),
            section=e.get("section", ""), op="registry", expr=None,
            locator=None, note=e.get("note", ""),
            sources=[dict(path=_rel(SPEC), sha16=sha16(SPEC))],
        )
        values[key] = val
        known[key] = val
        src_map[key] = [SPEC]

    manifest = hashlib.sha256(
        "\n".join(f"{k}\t{values[k]!r}" for k in sorted(values)).encode()
    ).hexdigest()[:16]

    payload = dict(
        registry_version=spec.get("version", 1),
        built_at=(built_at or datetime.now(timezone.utc).isoformat(timespec="seconds")),
        registry_file=os.path.basename(spec_path),
        registry_sha=sha16(spec_path),
        counts=dict(total=len(out),
                    main=sum(1 for v in out.values() if v["level"] == "main"),
                    literal=sum(1 for v in out.values() if v["op"] == "literal")),
        manifest_sha=manifest,
        problems=problems,
        entries=out,
    )

    if write:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "numbers.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        with open(os.path.join(out_dir, "numbers.md"), "w", encoding="utf-8") as f:
            f.write(to_markdown(payload))
        with open(os.path.join(out_dir, "numbers.tex"), "w", encoding="utf-8") as f:
            f.write(to_latex(payload))
    return payload


def to_markdown(payload):
    L = ["# 论文数字登记表", "",
         f"由 `scripts/build_registry.py` 从 `registry/numbers.yaml` 生成，**请勿手改**。", "",
         f"- 条目数：{payload['counts']['total']}（main {payload['counts']['main']}，"
         f"literal {payload['counts']['literal']}）",
         f"- 登记表哈希：`{payload['registry_sha']}`　取值清单哈希：`{payload['manifest_sha']}`",
         f"- 构建时间：{payload['built_at']}", ""]
    if payload["problems"]:
        L += ["## 构建告警", ""] + [f"- {p}" for p in payload["problems"]] + [""]
    cur = None
    for k, v in payload["entries"].items():
        if v["section"] != cur:
            cur = v["section"]
            L += ["", f"## {cur or '（未分组）'}", "",
                  "| 键 | 值 | 含义 | 文档 | 级别 |", "|---|---|---|---|---|"]
        L.append(f"| `{k}` | **{v['text']}** | {v['claim']} | {v['doc']} | {v['level']} |")
    L.append("")
    return "\n".join(L)


def latex_command(key):
    """LaTeX 控制词只能含字母，且不能含数字（\\CSzL21 会被切成 \\CSzL + 21）。"""
    words = [w for w in key.replace("-", "_").split("_") if w]
    out = "Reg" + "".join(w[:1].upper() + w[1:] for w in words)
    return "".join(ch for ch in out if ch.isalpha())


def to_latex(payload):
    L = ["% 由 scripts/build_registry.py 生成，勿手改",
         f"% registry_sha={payload['registry_sha']} manifest_sha={payload['manifest_sha']}"]
    for k, v in payload["entries"].items():
        L.append(f"\\newcommand{{\\{latex_command(k)}}}{{{v['text']}}}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="构建论文数字登记表")
    ap.add_argument("--spec", default=SPEC, help="登记表 yaml 路径")
    ap.add_argument("--out", default=os.environ.get("REGISTRY_OUT", REG),
                    help="产物目录")
    a = ap.parse_args()
    p = build(spec_path=a.spec, out_dir=a.out)
    print(f"条目 {p['counts']['total']}（main {p['counts']['main']}，"
          f"literal {p['counts']['literal']}）")
    print(f"registry_sha={p['registry_sha']} manifest_sha={p['manifest_sha']}")
    if p["problems"]:
        print("\n告警：")
        for x in p["problems"]:
            print("  -", x)
    else:
        print("无告警")
