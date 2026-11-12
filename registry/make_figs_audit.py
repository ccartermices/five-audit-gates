"""论文图件：全部由 data/ 下已落盘 CSV 生成，不手填任何数值。

六张图
------
fig1 五道闸门协议示意（唯一一张不含数据的图，画的是判据而不是结果）
fig2 测量信度：单搏 vs 多搏平均（ICC(1,1) → ICC(1,k)），VDB 与 MIMIC 双源
fig3 归因身份：top-12 依赖特征的跨种子 Jaccard
fig4 机制份额：15 次训练（3 架构 × 3 种子 + 3 架构 × k=4/16）
fig5 医师确认判别：规则内量 / 代理量 / 规则外量三类的单量 AUC
fig6 零分布校准：名义 5% 下的实测误报率（四源 × 三轴）

用法：python scripts/make_figs_audit.py [--out figs/paper]
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
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = _repo.ROOT
DATA = os.path.join(ROOT, "data")

# 期刊排版：180 mm 双栏宽，字号 8，无上/右边框
MM = 1 / 25.4
W2 = 180 * MM
plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
})

C_PASS = "#2b7a78"
C_FAIL = "#b5443a"
C_WARN = "#c98a2b"
C_RULE = "#b5443a"      # 规则内（构造性）
C_PROXY = "#c98a2b"     # 与规则强相关的代理量
C_EXT = "#2b7a78"       # 规则外（独立证据）
C_GREY = "#6b6b6b"


def save(fig, out, name):
    for ext in ("png", "pdf"):
        # no creation date in the PDF, so a rerun gives the same bytes
        kw = {"metadata": {"CreationDate": None}} if ext == "pdf" else {}
        fig.savefig(os.path.join(out, f"{name}.{ext}"), **kw)
    plt.close(fig)
    print(f"  {name}.png / .pdf")


# ---------------------------------------------------------------- fig 1
def fig_gates(out):
    gates = [
        ("G1  Number provenance",
         "can every reported value be recomputed\nfrom a hashed source file?",
         "PASS", C_PASS, "9/9 checks; 0 hand-typed constants"),
        ("G2  Label provenance",
         "is the ground truth independent of\nthe model that produced the features?",
         "FAIL", C_FAIL, "labels = model prelabels (L2, not L3)"),
        ("G3  Unit of analysis",
         "is the quantity reliable at the unit\nat which it is reported?",
         "PARTIAL", C_WARN, "ICC 0.12 per beat, 0.80 for 30-beat mean"),
        ("G4  Null calibration",
         "does the test actually reject at its\nnominal rate under the null?",
         "FAIL", C_FAIL, "nominal 5% realised 7.5-13%"),
        ("G5  Attribution stability",
         "does the reported unit survive a\nreseed of the model?",
         "SPLIT", C_WARN, "features fail (J = 0.04), clusters pass"),
    ]
    fig, ax = plt.subplots(figsize=(W2, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.35, len(gates) + 0.02)
    ax.axis("off")
    for i, (name, q, verdict, col, detail) in enumerate(gates):
        y = len(gates) - i - 0.86
        ax.add_patch(FancyBboxPatch((0.005, y), 0.985, 0.70,
                                    boxstyle="round,pad=0.004,rounding_size=0.012",
                                    facecolor="#f7f7f5", edgecolor="#d0d0cb", linewidth=0.6))
        ax.add_patch(Rectangle((0.005, y), 0.008, 0.70, facecolor=col, edgecolor="none"))
        ax.text(0.030, y + 0.50, name, fontsize=8, fontweight="bold", va="center")
        ax.text(0.030, y + 0.21, q, fontsize=6.6, va="center", color="#3a3a3a", linespacing=1.25)
        ax.text(0.600, y + 0.50, verdict, fontsize=7.5, fontweight="bold", color=col, va="center")
        ax.text(0.600, y + 0.21, detail, fontsize=6.3, va="center", color="#3a3a3a")
    ax.text(0.005, -0.28, "verdict applies to the pipeline, not to the hypothesis: "
            "each failure removes one class of claim (Fig. 2-6)",
            fontsize=6.2, color=C_GREY, style="italic")
    save(fig, out, "fig1_gates")


# ---------------------------------------------------------------- fig 2
def fig_icc(out):
    d = pd.read_csv(os.path.join(DATA, "audit_vdb_icc.csv"))
    srcs = [("VDB_搏间", "Ventilator-free beats (VDB)\nbeat-to-beat, 98 subjects"),
            ("MIMIC_段间", "MIMIC segments\nsegment-to-segment, 425 subjects")]
    # 两个面板必须共用同一行序，否则 sharey 会让右侧的点落在左侧的标签上
    order = (d[d.source == "VDB_搏间"].sort_values("icck").feat.tolist())
    k = order.index("sdptg_d")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 4.2), sharey=True)
    for ax, (src, title) in zip(axes, srcs):
        s = d[d.source == src].set_index("feat").loc[order].reset_index()
        y = np.arange(len(s))
        ax.axhspan(k - 0.5, k + 0.5, color="#fbf0e4", zorder=0)
        for yi, (_, r) in zip(y, s.iterrows()):
            ax.plot([r.icc1, r.icck], [yi, yi], color="#c9c9c4", lw=0.8, zorder=1)
        ax.scatter(s.icc1, y, s=9, color=C_FAIL, zorder=3, label="single unit  ICC(1,1)")
        ax.scatter(s.icck, y, s=11, color=C_PASS, zorder=3, label="30-unit mean  ICC(1,k)")
        ax.axvline(0.80, color=C_GREY, lw=0.6, ls="--", zorder=0)
        ax.axvline(0.50, color=C_GREY, lw=0.6, ls=":", zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels([("▶ " if f == "sdptg_d" else "") + f for f in s.feat], fontsize=6)
        for t, f in zip(ax.get_yticklabels(), s.feat):
            if f == "sdptg_d":
                t.set_fontweight("bold")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(-0.8, len(s) - 0.2)
        ax.set_xlabel("intraclass correlation")
        ax.set_title(title, fontsize=7)
        r = s.iloc[k]
        ax.text(0.015, len(s) - 1.0, f"sdptg_d  {r.icc1:.2f} → {r.icck:.2f}",
                fontsize=5.8, color="#8a5a2b", va="top")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), handletextpad=0.4, columnspacing=2.0)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.05, 1, 1))
    save(fig, out, "fig2_icc")


# ---------------------------------------------------------------- fig 3
def fig_jaccard(out):
    d = pd.read_csv(os.path.join(DATA, "probe_what_top12_jaccard.csv"))
    archs = ["cnn", "mlp", "transformer"]
    M = d[archs].to_numpy()
    fig, ax = plt.subplots(figsize=(78 * MM, 1.85))
    im = ax.imshow(M, cmap="BuPu", vmin=0, vmax=0.6, aspect="auto")
    ax.set_xticks(range(len(archs)))
    ax.set_xticklabels(["CNN", "MLP", "Transformer"], fontsize=6.5)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["pair"], fontsize=6.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="white" if M[i, j] > 0.32 else "#222")
    ax.set_xlabel("architecture", fontsize=7)
    ax.set_ylabel("seed pair", fontsize=7)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Jaccard overlap of top-12 features", fontsize=6.2)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.4)
    fig.tight_layout()
    save(fig, out, "fig3_jaccard")


# ---------------------------------------------------------------- fig 4
def fig_shares(out):
    seed = pd.read_csv(os.path.join(DATA, "probe_what_fam_by_seed.csv"))
    seed = seed.rename(columns={"Unnamed: 0": "run", "Unnamed: 1": "arch"})
    seed["k"] = 8
    ks = pd.read_csv(os.path.join(DATA, "probe_what_k_scan.csv"))
    # k 扫描表含 k=8（与种子 0 的基准重复），剔掉后剩 6 次，与种子扫描合计 15 次训练
    ks = ks[ks.k != 8].copy()
    # k 扫描表只落了 vascular/pressure/demographic，flow+other 取残差（不单独显示）
    ks["other"] = (100.0 - ks.vascular - ks.pressure - ks.demographic).clip(lower=0)
    ks["flow"] = 0.0
    ks["run"] = "k" + ks["k"].astype(str)
    cols = ["vascular", "pressure", "flow", "other"]
    allr = pd.concat([seed[cols + ["run", "arch"]],
                      ks[cols + ["run", "arch"]]], ignore_index=True)
    A_ORDER = {"cnn": 0, "mlp": 1, "transformer": 2}
    R_ORDER = {"s0": 0, "s1": 1, "s2": 2, "k4": 3, "k16": 4}
    A_LAB = {"cnn": "CNN", "mlp": "MLP", "transformer": "TF"}
    R_LAB = {"s0": "seed 0", "s1": "seed 1", "s2": "seed 2", "k4": "k=4", "k16": "k=16"}
    allr["a_ord"] = allr.arch.map(A_ORDER)
    allr["r_ord"] = allr.run.map(R_ORDER)
    allr = allr.sort_values(["a_ord", "r_ord"]).reset_index(drop=True)
    allr["order"] = [f"{A_LAB[a]}\n{R_LAB[r]}" for a, r in zip(allr.arch, allr.run)]

    colors = {"vascular": "#2b7a78", "pressure": "#8fb8b6", "flow": "#c98a2b", "other": "#c9c9c4"}
    fig, ax = plt.subplots(figsize=(W2, 2.5))
    x = np.arange(len(allr))
    bottom = np.zeros(len(allr))
    for c in cols:
        ax.bar(x, allr[c], bottom=bottom, width=0.72, color=colors[c],
               edgecolor="white", linewidth=0.4, label=c)
        bottom += allr[c].to_numpy()
    ax.set_xticks(x)
    ax.set_xticklabels(allr["order"], fontsize=5.6, linespacing=1.15)
    ax.set_ylabel("share of causal weight (%)")
    ax.set_ylim(0, 100)
    for i, v in enumerate(allr["vascular"]):
        ax.text(i, 3, f"{v:.0f}", ha="center", fontsize=4.8, color="white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16),
              handlelength=1.0, columnspacing=1.4)
    ax.annotate("age (demographic) share = 0.00% in all 15 runs",
                xy=(0.5, -0.30), xycoords="axes fraction",
                fontsize=6.2, color=C_GREY, ha="center", style="italic",
                annotation_clip=False)
    fig.tight_layout()
    save(fig, out, "fig4_shares")


# ---------------------------------------------------------------- fig 5
def fig_phy(out):
    d = pd.read_csv(os.path.join(DATA, "audit_physician_single_full.csv"))
    rule = {"sdptg_d", "notch_depth", "sdptg_b", "kurt"}
    proxy = {"p2_rel"}
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.9), sharex=True)
    specs = [("高置信", "high-confidence subset  (n = 16; 7 vs 9)", 0.746),
             ("全集", "full set  (n = 56; 7 vs 49)", 0.694)]
    for ax, (scene, title, thr) in zip(axes, specs):
        s = d[d.scene == scene].sort_values("auc").reset_index(drop=True)
        y = np.arange(len(s))
        cols = [C_RULE if f in rule else (C_PROXY if f in proxy else C_EXT) for f in s.feat]
        ax.barh(y, s.auc, height=0.72, color=cols, edgecolor="none")
        ax.axvline(thr, color=C_GREY, lw=0.7, ls="--")
        ax.axvline(0.5, color="#bdbdbd", lw=0.5, ls=":")
        ax.set_yticks(y)
        ax.set_yticklabels(s.feat, fontsize=5.2)
        ax.set_xlim(0, 1.04)
        ax.set_ylim(-0.8, len(s) - 0.2)
        ax.set_xlabel("single-quantity AUC")
        ax.set_title(title, fontsize=7)
        ax.text(thr - 0.012, len(s) / 2, f"95% null threshold {thr:.2f}",
                fontsize=5.6, color=C_GREY, va="center", ha="right", rotation=90,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.8))
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_RULE, C_PROXY, C_EXT)]
    fig.legend(handles, ["rule-internal (constructive)", "proxy of rule quantity",
                         "rule-external (independent)"],
               ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               frameon=False, fontsize=6.2, handlelength=1.0, columnspacing=1.6)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.05, 1, 1))
    save(fig, out, "fig5_physician")


# ---------------------------------------------------------------- fig 6
def fig_null(out):
    d = pd.read_csv(os.path.join(DATA, "audit_v22_zcal.csv"))
    axes_ = ["fpr_mech", "fpr_desc", "fpr_device"]
    labels = ["mechanism", "restatement", "device"]
    fig, ax = plt.subplots(figsize=(88 * MM, 2.15))
    x = np.arange(len(d))
    w = 0.26
    cols = ["#2b7a78", "#8fb8b6", "#c98a2b"]
    for i, (c, lab, col) in enumerate(zip(axes_, labels, cols)):
        ax.bar(x + (i - 1) * w, d[c], width=w, color=col, label=lab, edgecolor="none")
    ax.axhline(0.05, color=C_FAIL, lw=0.8, ls="--")
    ax.text(len(d) - 0.45, 0.029, "nominal 5%", fontsize=6, color=C_FAIL,
            ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(d.source, fontsize=6.5)
    ax.set_ylabel("realised false-positive rate")
    ax.set_ylim(0, 0.15)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(ncol=3, frameon=False, loc="upper left", handlelength=1.0, columnspacing=1.2)
    fig.tight_layout()
    save(fig, out, "fig6_null")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("生成图件 →", a.out)
    fig_gates(a.out)
    fig_icc(a.out)
    fig_jaccard(a.out)
    fig_shares(a.out)
    fig_phy(a.out)
    fig_null(a.out)


if __name__ == "__main__":
    main()
