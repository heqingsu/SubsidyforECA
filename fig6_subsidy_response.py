# -*- coding: utf-8 -*-
"""Redraw Figure 4 (exp04_subsidy) from the archived subsidy-response table.
Panel letters are placed inside each axes; no title floats above a panel."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path(r"F:/sharefile/碳论文/Energies-4678462/experiment")
spec = importlib.util.spec_from_file_location("rf", EXP / "revision_final.py")
rf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rf)
BUDGET = rf.C().budget

base = pd.read_csv(EXP / "outputs/tables/subsidy_response.csv")
good = base.budget_feasible.astype(bool) if "budget_feasible" in base else base.spending <= BUDGET

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                     "axes.labelsize": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
fig, ax = plt.subplots(2, 2, figsize=(10.2, 7.2), sharex=True)

ax[0, 0].plot(base.subsidy, base.price, color="#1f4e79", label="Service price")
ax[0, 0].plot(base.subsidy, base.effective_price, color="#c0504d", ls="--",
              label="Effective payment")
ax[0, 0].set_ylabel("CNY per enterprise")
ax[0, 0].legend(frameon=False, loc="lower left", fontsize=8.5)

ax[0, 1].plot(base.subsidy, base.participation * 100, color="#1f4e79", label="With subsidy")
ax[0, 1].plot(base.subsidy, base.withdrawal * 100, color="#c0504d", ls="--",
              label="After withdrawal")
ax[0, 1].set_ylabel("Participation (%)")
ax[0, 1].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.88), fontsize=8.5)

ax[1, 0].plot(base.subsidy, base.spending / 1e6, color="#1f4e79")
ax[1, 0].axhline(BUDGET / 1e6, color="#7f7f7f", ls=":", label="Budget")
ax[1, 0].set_ylabel("Expected expenditure\n(million CNY)")
ax[1, 0].legend(frameon=False, loc="lower right", fontsize=8.5)

ax[1, 1].plot(base.subsidy, np.where(good, base.government, np.nan),
              color="#1f4e79", label="Government score")
ax[1, 1].plot(base.subsidy, np.where(good, base.exceed_risk, np.nan) * 100, "--",
              color="#c0504d", label="Overspend risk (%)")
ax[1, 1].set_ylabel("Score / risk (%)")
ax[1, 1].legend(frameon=False, loc="upper right", fontsize=8.5)

for a, lab in zip(ax.flat, ["(a)", "(b)", "(c)", "(d)"]):
    a.text(0.02, 0.96, lab, transform=a.transAxes, ha="left", va="top",
           fontsize=10, fontweight="bold", color="#333333")
    a.set_xlabel("Subsidy (CNY per adopter)")
    a.grid(axis="y", color="#e6e6e6", lw=0.6)
    a.set_axisbelow(True)

fig.tight_layout(h_pad=1.4)
out = EXP / "outputs/figures"
(out / "exp04_subsidy.pdf").parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "exp04_subsidy.pdf", bbox_inches="tight")
fig.savefig(out / "exp04_subsidy.png", dpi=200, bbox_inches="tight")
print("saved", out / "exp04_subsidy.pdf")
