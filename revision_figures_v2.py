"""Regenerate Figures 3 and 10 in response to Reviewer 1, comment 7.

Figure 3  : larger configuration labels, wider panels, legend-free layout.
Figure 10 : discrete colour blocks with a grouped legend instead of in-cell
            text labels, so that the selection pattern is legible at print size.

Outputs overwrite outputs/figures/exp02_aggregation.{pdf,png} and
outputs/figures/final_resolution_map.{pdf,png}; the submitted versions are kept
in outputs/figures_submitted/.
"""
from pathlib import Path
import importlib.util
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
FIG = HERE / "outputs" / "figures"
TAB = HERE / "outputs" / "tables"
SUB = HERE / "outputs" / "figures_submitted"
SUB.mkdir(parents=True, exist_ok=True)
for name in ("exp02_aggregation.pdf", "exp02_aggregation.png",
             "final_resolution_map.pdf", "final_resolution_map.png"):
    src = FIG / name
    if src.exists() and not (SUB / name).exists():
        shutil.copy2(src, SUB / name)

spec = importlib.util.spec_from_file_location("rf", HERE / "revision_final.py")
rf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rf)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
})

ORDER = ["t24h_g1", "t24h_g4", "t24h_g8", "t6h_g1", "t6h_g4", "t6h_g8", "t1h_g1", "t1h_g4", "t1h_g8"]
SHORT = {"t24h_g1": "24h/1z", "t24h_g4": "24h/4z", "t24h_g8": "24h/8z",
         "t6h_g1": "6h/1z", "t6h_g4": "6h/4z", "t6h_g8": "6h/8z",
         "t1h_g1": "1h/1z", "t1h_g4": "1h/4z", "t1h_g8": "1h/8z",
         "no_service": "no service"}

# ---------------------------------------------------------------- Figure 3
base = rf.C()
D = rf.build(rf.E0, rf.phi0, rf.w0, base)
agg = pd.read_csv(TAB / "aggregation.csv", encoding="utf-8-sig").set_index("configuration")

plt.rcParams.update({"xtick.labelsize": 11, "ytick.labelsize": 10.5,
                     "axes.labelsize": 11.5, "axes.titlesize": 11.5})
fig, ax = plt.subplots(1, 3, figsize=(14.6, 4.8))
bp = ax[0].boxplot([D[n]["D"] for n in ORDER], widths=0.55, showfliers=False,
                   patch_artist=True, medianprops=dict(color="#08306b", lw=1.2))
for box in bp["boxes"]:
    box.set(facecolor="#c9d9ec", edgecolor="#3978a8", lw=0.8)
ax[0].axhline(0, color="k", lw=0.7)
ax[0].set_xticklabels([SHORT[n] for n in ORDER], rotation=45, ha="right")
ax[0].set_ylabel("Reported minus refined (tCO$_2$e)")
ax[0].set_title("(a) Signed accounting differences")

ax[1].bar([SHORT[n] for n in ORDER], 100 * agg.loc[ORDER, "weighted_deviation"],
          color="#3978a8", width=0.62)
ax[1].set_xticklabels([SHORT[n] for n in ORDER], rotation=45, ha="right")
ax[1].set_ylabel("Weighted deviation (%)")
ax[1].set_title("(b) Deviation from the refined benchmark")

ax[2].bar([SHORT[n] for n in ORDER], agg.loc[ORDER, "fixed_cost"] / 1000,
          color="#6b8fb5", width=0.62)
ax[2].set_xticklabels([SHORT[n] for n in ORDER], rotation=45, ha="right")
ax[2].set_ylabel("Fixed cost (thousand CNY/year)")
ax[2].set_title("(c) Annual configuration cost")

for a in ax:
    a.tick_params(axis="x", labelsize=11, pad=2)
    a.tick_params(axis="y", labelsize=10.5)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"exp02_aggregation.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Figure 3 regenerated")

# --------------------------------------------------------------- Figure 10
COLORS = {
    "t1h_g8": "#08306b", "t1h_g4": "#2171b5", "t1h_g1": "#6baed6",
    "t6h_g8": "#00441b", "t6h_g4": "#238b45", "t6h_g1": "#74c476",
    "t24h_g8": "#7f2704", "t24h_g4": "#d94801", "t24h_g1": "#fd8d3c",
    "no_service": "#d9d9d9",
}
CODES = {n: i for i, n in enumerate(ORDER + ["no_service"])}
cmap = ListedColormap([COLORS[n] for n in ORDER + ["no_service"]])

maps = [("(a) Baseline factor profile, V$_0$ = 2,000 CNY", pd.read_csv(TAB / "final_map_compliance_value.csv", encoding="utf-8-sig", index_col=0)),
        ("(b) Flattened factor profile, V$_0$ = 2,000 CNY", pd.read_csv(TAB / "final_map_flat_profile.csv", encoding="utf-8-sig", index_col=0))]

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
for ax, (title, grid) in zip(axes, maps):
    g = grid.apply(lambda col: col.map(lambda x: CODES.get(x, 9))).to_numpy(int)
    ax.imshow(g, aspect="auto", cmap=cmap, vmin=0, vmax=9, origin="lower", interpolation="nearest")
    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels([c[1:] for c in grid.columns], fontsize=7.5)
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels([i[1:] for i in grid.index], fontsize=7.5)
    ax.set_xlabel("Recognition rate $\\eta$", fontsize=8)
    ax.set_ylabel("Cost multiplier", fontsize=8)
    ax.set_title(title, fontsize=8.5)
    ax.set_xticks(np.arange(-.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.6)
    ax.tick_params(which="minor", length=0)

handles = [Patch(facecolor=COLORS[n], edgecolor="none", label=SHORT[n]) for n in ORDER]
handles.append(Patch(facecolor=COLORS["no_service"], edgecolor="none", label="no service"))
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
           bbox_to_anchor=(0.5, -0.06), fontsize=7.5, handlelength=1.1, columnspacing=1.2)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"final_resolution_map.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Figure 10 regenerated")
