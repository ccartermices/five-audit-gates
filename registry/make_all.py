"""Regenerate every figure in the paper, in order.

    python scripts/05_figures/make_all.py

Portable entry point, so the repository does not depend on make being installed. Reads
data/*.csv only and overwrites figures/ in place.
"""
from __future__ import annotations

import os as _os, sys as _sys

_S = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _S)
for _d in _os.listdir(_S):
    if _d[:1].isdigit() and _os.path.isdir(_os.path.join(_S, _d)):
        _sys.path.append(_os.path.join(_S, _d))
import _repo  # noqa: E402

import subprocess

HERE = _os.path.join(_repo.SCRIPTS, "05_figures")

STEPS = [
    ("fig1_gates, fig2_icc, fig3_jaccard, fig4_shares, fig5_physician, fig6_null",
     ["make_figs_audit.py"]),
    ("fig7_dose, fig8_profile, fig9_onedim, fig10_taxonomy",
     ["make_figs_interp.py"]),
    ("fig3_sae_attrib", ["make_fig_jbhi.py"]),
]


def main() -> None:
    for label, cmd in STEPS:
        print(f"\n=== {label}")
        p = subprocess.run([_sys.executable] + cmd, cwd=HERE)
        if p.returncode != 0:
            raise SystemExit(f"{cmd[0]} failed with exit code {p.returncode}")
    print(f"\nfigures written to {_repo.FIGURES}")


if __name__ == "__main__":
    main()
