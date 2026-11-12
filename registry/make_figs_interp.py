# -*- coding: utf-8 -*-
"""可解释性章的四张图 —— 全部由 data/ 下 CSV 生成，不手填数值。

fig7  剂量-反应：五个形态算子 × 三架构，输出变化随剂量
fig8  位移剖面：算子 × L1 簇（PWDB 与 CM 双面板），噪声算子应孤立在 device
fig9  一维性：简约曲线（左）与整簇消融（右）
fig10 两套命名体系下同一批特征的归属（仿真器真值族 × L1 簇）

用法：python scripts/make_figs_interp.py [--out figs/paper]
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

ROOT = _repo.ROOT
DATA = os.path.join(ROOT, "data")

MM = 1 / 25.4
W2 = 180 * MM
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "savefig.bbox": "tight", "savefig.dpi": 300, "font.family": "DejaVu Sans",
})

C_PASS = "#2b7a78"
C_FAIL = "#b5443a"
C_WARN = "#c98a2b"
C_GREY = "#6b6b6b"
ARCH_C = {"cnn": "#2b7a78", "mlp": "#c98a2b", "transformer": "#4a5b8c"}
ARCH_L = {"cnn": "CNN", "mlp": "MLP", "transformer": "Transformer"}

ORDER = ["notch", "late", "reflect", "upstroke", "noise"]
LABEL = {"notch": "notch", "late": "late wave", "reflect": "reflection P2",
         "upstroke": "upstroke", "noise": "HF noise (control)"}


def save(fig, out, name):
    for ext in ("png", "pdf"):
        # no creation date in the PDF, so a rerun gives the same bytes
        kw = {"metadata": {"CreationDate": None}} if ext == "pdf" else {}
        fig.savefig(os.path.join(out, f"{name}.{ext}"), **kw)
    plt.close(fig)
    print(f"  {name}.png / .pdf")


def fig_dose(out):
    d = pd.read_csv(os.path.join(DATA, "xinterp_dose.csv"))
    fig, axes = plt.subplots(1, 5, figsize=(W2, 2.45), sharey=True)
    for ax, op in zip(axes, ORDER):
        for arch in ["cnn", "mlp", "transformer"]:
            s = d[(d.op == op) & (d.arch == arch)].sort_values("dose")
            ax.plot(s.dose, s.mean_dpred, "-o", ms=2.4, lw=1.0,
                    color=ARCH_C[arch], label=ARCH_L[arch])
            ax.fill_between(s.dose, s.lo, s.hi, color=ARCH_C[arch], alpha=0.12,
                            lw=0)
        ax.axhline(0, color="#c9c9c4", lw=0.6, zorder=0)
        ax.axvline(0, color="#c9c9c4", lw=0.6, zorder=0)
        ax.set_title(LABEL[op], fontsize=6.8)
        ax.set_xlabel("edit amplitude")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Δ predicted pressure\n(SD of outcome)")
    axes[0].set_ylim(-0.60, 0.52)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 1.005), handlelength=1.4, fontsize=7)
    axes[4].text(0.002, 0.48, "only the transformer\nfollows it", fontsize=5.8,
                 color=C_FAIL, va="top")
    fig.tight_layout(w_pad=0.6)
    save(fig, out, "fig7_dose")


def fig_profile(out):
    pw = pd.read_csv(os.path.join(DATA, "xinterp_spec_summary.csv"))
    cm = pd.read_csv(os.path.join(DATA, "xinterp_spec_summary_cm.csv"))
    clusters = ["stiffness", "reflection", "dicrotic", "duration",
                "shape", "spectrum", "noise"]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.5), sharey=True)
    for ax, df, title in [(axes[0], pw, "PWDB (virtual cycles)"),
                          (axes[1], cm, "CM (real pulses)")]:
        M = np.array([[df[df.op == op][f"c_{c}"].values[0] for c in clusters]
                      for op in ORDER], float)
        VMX = 4.0
        im = ax.imshow(np.clip(M, 0, VMX), aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=VMX)
        ax.set_xticks(range(len(clusters)))
        ax.set_xticklabels(clusters, rotation=40, ha="right", fontsize=6)
        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels([LABEL[o] for o in ORDER], fontsize=6)
        ax.set_title(title, fontsize=7)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i, j] >= 1:
                    ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                            fontsize=5.2, fontweight="bold" if M[i, j] > VMX else "normal",
                            color="white" if M[i, j] > VMX * 0.55 else "#222222")
        ax.axhline(3.5, color="#333333", lw=0.8)
        ax.set_xlabel("L1 cluster displaced")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("mean |Δ| (population SD, clipped at %.0f)" % VMX, fontsize=6)
    cb.ax.tick_params(labelsize=6)
    save(fig, out, "fig8_profile")


def fig_onedim(out):
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.6))
    ax = axes[0]
    for arch in ["cnn", "mlp", "transformer"]:
        p = pd.read_csv(os.path.join(DATA, f"xinterp_parsimony_{arch}.csv"))
        ax.plot(p.k, p.rho, "-o", ms=2.2, lw=1.0, color=ARCH_C[arch],
                label=ARCH_L[arch])
    ax.annotate("k = 1 already reaches 0.96–0.99", xy=(1.0, 0.9585),
                xytext=(6.0, 0.9545), fontsize=5.8, color=C_GREY,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=C_GREY))
    ax.set_xlabel("number of L1 quantities in the surrogate")
    ax.set_ylabel("ρ with the black box's output")
    ax.set_ylim(0.950, 1.001)
    ax.set_title("sufficiency", fontsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=6)

    ax = axes[1]
    n = pd.read_csv(os.path.join(DATA, "xinterp_necessity.csv"))
    k = int(n.n_feat_in_cluster.min())
    k = min(2, k)
    s = n[n.k == k]
    clusters = ["stiffness", "reflection", "dicrotic", "duration",
                "shape", "spectrum", "noise"]
    w = 0.26
    for i, arch in enumerate(["cnn", "mlp", "transformer"]):
        vals = [float(s[(s.arch == arch) & (s.cluster == c)].mean_abs_dpred
                      .values[0]) if len(s[(s.arch == arch) &
                                           (s.cluster == c)]) else 0.0
                for c in clusters]
        ax.bar(np.arange(len(clusters)) + (i - 1) * w, vals, width=w,
               color=ARCH_C[arch], label=ARCH_L[arch], lw=0)
    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels(clusters, rotation=40, ha="right", fontsize=6)
    ax.set_ylabel("|Δ prediction| (SD)")
    ax.set_title(f"necessity: remove {k} features of a cluster", fontsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=6)
    ax.text(0.02, 0.97, "descriptor clusters carry the output",
            transform=ax.transAxes, fontsize=5.8, color=C_GREY, va="top")
    fig.tight_layout(w_pad=1.4)
    save(fig, out, "fig9_onedim")


def fig_taxonomy(out):
    d = pd.read_csv(os.path.join(DATA, "xinterp_taxonomy_cross.csv"))
    fams = ["vascular", "pressure", "flow"]
    cls = ["stiffness", "reflection", "dicrotic", "duration", "shape",
           "spectrum", "noise"]
    M = np.zeros((len(fams), len(cls)))
    for _, r in d.iterrows():
        if r.truth_family in fams and r.l1_cluster in cls:
            M[fams.index(r.truth_family), cls.index(r.l1_cluster)] += r.d_pred
    M = 100 * M / M.sum()
    fig, ax = plt.subplots(figsize=(W2 * 0.62, 2.0))
    im = ax.imshow(M, aspect="auto", cmap="YlGnBu", vmin=0, vmax=M.max())
    ax.set_xticks(range(len(cls)))
    ax.set_xticklabels(cls, rotation=35, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels([f"truth: {f}" for f in fams], fontsize=6.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] >= 0.5:
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                        fontsize=6, color="white" if M[i, j] > M.max() * 0.6
                        else "#222222")
    ax.set_xlabel("L1 cluster (same features, second label set)", fontsize=7)
    ax.set_title("causal weight by label set (%)", fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    fig.tight_layout()
    save(fig, out, "fig10_taxonomy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "figures"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_dose(args.out)
    fig_profile(args.out)
    fig_onedim(args.out)
    fig_taxonomy(args.out)


if __name__ == "__main__":
    main()
