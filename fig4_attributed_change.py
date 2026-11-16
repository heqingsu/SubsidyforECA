# -*- coding: utf-8 -*-
"""Regenerate manuscript Figure 4 (rev04_attributed_change): attribution capability
and cost of measurement. The 'no intra-day shift' note is placed below the bars,
not on top of them; panel letters inside the axes."""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path(r"F:/sharefile/碳论文/Energies-4678462/experiment")
OUTF = EXP / "outputs/figures"
att = pd.read_csv(EXP / "outputs/tables/final_attributed_change.csv")

order = ["t24h_g1", "t24h_g4", "t24h_g8", "t6h_g1", "t6h_g4", "t6h_g8",
         "t1h_g1", "t1h_g4", "t1h_g8"]
labels = {"t24h_g1": "24h/1z", "t24h_g4": "24h/4z", "t24h_g8": "24h/8z",
          "t6h_g1": "6h/1z", "t6h_g4": "6h/4z", "t6h_g8": "6h/8z",
          "t1h_g1": "1h/1z", "t1h_g4": "1h/4z", "t1h_g8": "1h/8z"}
o = att.set_index("configuration").loc[order].reset_index()
y = np.arange(len(order))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                     "axes.labelsize": 10, "axes.titlesize": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})
fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.8), sharey=True)

# (a) attribution capability
vals = o.attributed_tCO2e / 1000.0
ax[0].barh(y, vals, color="#2f6f5e", height=0.62)
ax[0].set_yticks(y, [labels[c] for c in order])
ax[0].set_xlabel("Attributed accounting change (thousand tCO$_2$e/year)")
ax[0].axvline(0, color="k", lw=0.8)
ax[0].grid(axis="x", color="#e6e6e6", lw=0.6); ax[0].set_axisbelow(True)

# (b) cost of measurement; guard against the near-zero/NaN ratios of 24h configs
cp = o.cost_per_attributed_tonne.to_numpy(float)
plot_cp = np.where((cp > 0) & (cp < 100), cp, np.nan)
ax[1].barh(y, np.nan_to_num(plot_cp), color="#a4552f", height=0.62)
ax[1].set_xlabel("Configuration cost per attributed tonne (CNY/tCO$_2$e)")
ax[1].grid(axis="x", color="#e6e6e6", lw=0.6); ax[1].set_axisbelow(True)

# note as a caption-style footnote below panel (b), never on the bars
ax[1].text(0.0, -0.30,
           "24h configurations attribute no intra-day shift; their cost per\n"
           "attributed tonne is undefined and is left blank.",
           transform=ax[1].transAxes, ha="left", va="top", fontsize=7.5,
           color="#555555")

for a, lab in zip(ax, ["(a)", "(b)"]):
    a.text(0.02, 0.97, lab, transform=a.transAxes, ha="left", va="top",
           fontsize=10, fontweight="bold", color="#333333",
           bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2))

ax[0].invert_yaxis()
fig.tight_layout()
fig.subplots_adjust(bottom=0.30)
fig.savefig(OUTF / "rev04_attributed_change.pdf", bbox_inches="tight")
fig.savefig(OUTF / "rev04_attributed_change.png", dpi=200, bbox_inches="tight")
print("saved", OUTF / "rev04_attributed_change.pdf")
