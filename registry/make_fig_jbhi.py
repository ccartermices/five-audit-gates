# -*- coding: utf-8 -*-
"""JBHI 版的合并图：fig3_sae_attrib（两栏图）与 fig1 的图注扩容。

为什么合并
----------
JBHI 的免费线是 8 页。10 图 5 表按原样放不下，所以按修改建议 §5.3 把
「特征身份不随重种子存活」与「机制份额只在标签体系内存活」合成一张两栏图：
两者讲的是同一件事的两面——归因的**单位**在哪里失效、在哪里幸存。

原 fig3（jaccard）与 fig4（shares）仍然单独保留在 data 与 figs 里，
以备审稿人索要单栏版本；本脚本只负责生成合并图。

用法：python scripts/make_fig_jbhi.py
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


import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = _repo.ROOT
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figures")

MM = 1 / 25.4
W2 = 180 * MM          # 双栏宽 180 mm
H2 = 108 * MM          # 双栏图高上限（IEEE 单页可容 108 mm）

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

C_VASC = "#2b7a78"
C_PRESS = "#8fb8b6"
C_OTHER = "#c9c9c4"
C_GREY = "#6b6b6b"


def save(fig, name):
    for ext in ("png", "pdf"):
        # no creation date in the PDF, so a rerun gives the same bytes
        kw = {"metadata": {"CreationDate": None}} if ext == "pdf" else {}
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), **kw)
    plt.close(fig)
    print(f"  {name}.png / .pdf")


def panel_jaccard(ax):
    """左：top-12 依赖特征的跨种子 Jaccard。"""
    d = pd.read_csv(os.path.join(DATA, "probe_what_top12_jaccard.csv"))
    archs = ["cnn", "mlp", "transformer"]
    M = d[archs].to_numpy()
    im = ax.imshow(M, cmap="BuPu", vmin=0, vmax=0.6, aspect="auto")
    ax.set_xticks(range(len(archs)))
    ax.set_xticklabels(["CNN", "MLP", "Transformer"])
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["pair"])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="white" if M[i, j] > 0.32 else "#222")
    ax.set_xlabel("architecture")
    ax.set_ylabel("seed pair")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("(a) feature identity across seeds", fontsize=8)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Jaccard overlap of top-12 features", fontsize=6.2)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.4)


def panel_shares(ax):
    """右：15 次训练里的机制份额堆叠柱。"""
    seed = pd.read_csv(os.path.join(DATA, "probe_what_fam_by_seed.csv"))
    seed = seed.rename(columns={"Unnamed: 0": "run", "Unnamed: 1": "arch"})
    seed["k"] = 8
    ks = pd.read_csv(os.path.join(DATA, "probe_what_k_scan.csv"))
    ks = ks[ks.k != 8].copy()
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

    x = np.arange(len(allr))
    bottom = np.zeros(len(allr))
    for c, col in zip(cols, [C_VASC, C_PRESS, "#c9c9c4", C_OTHER]):
        ax.bar(x, allr[c], bottom=bottom, width=0.72, color=col,
               edgecolor="white", linewidth=0.4, label=c)
        bottom += allr[c].to_numpy()
    ax.set_xticks(x)
    # 15 根柱挤在半栏宽里，两行标签会互相压字。改成「架构在上、运行在下」
    # 的错行排布：同一架构的 5 个运行标签轮流两行，相邻柱不共线。
    ax.set_xticklabels(allr["order"], fontsize=5.4, linespacing=1.0, rotation=90)
    ax.set_ylabel("share of causal weight (%)")
    ax.set_ylim(0, 100)
    for i, v in enumerate(allr["vascular"]):
        ax.text(i, 3, f"{v:.0f}", ha="center", fontsize=4.8, color="white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14),
              handlelength=1.0, columnspacing=1.4)
    ax.set_title("(b) mechanism share across 15 runs", fontsize=8)
    ax.annotate("age (demographic) share = 0.00% in all 15 runs",
                xy=(0.5, -0.40), xycoords="axes fraction",
                fontsize=6.2, color=C_GREY, ha="center", style="italic",
                annotation_clip=False)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(W2, H2),
                             gridspec_kw={"width_ratios": [1, 1.55]})
    panel_jaccard(axes[0])
    panel_shares(axes[1])
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig3_sae_attrib")
    print("[fig3_sae_attrib] 合并图已生成（原 fig3_jaccard + fig4_shares）")


if __name__ == "__main__":
    main()
