"""Regenerate the manuscript figures that depend on policy results, using the
revised baseline (cap 20,000, omega = 0). Settings-based figures exp01-exp03
are unchanged and reused from the original package.
"""
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import revision_experiments as R

FIG, TAB = R.FIG, R.TAB
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})
c0 = R.Settings()
data, _ = R.prepare(R.E0, c0)
base = pd.read_csv(TAB / "subsidy_response.csv")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---- exp04: subsidy response ------------------------------------------------
plt.rcParams.update({"font.size": 9.5, "axes.labelsize": 10, "axes.titlesize": 10})
fig, ax = plt.subplots(2, 2, figsize=(10.2, 7.2), sharex=True)
good = base.budget_feasible
ax[0, 0].plot(base.subsidy, base.price, color="#1f4e79", label="Service price")
ax[0, 0].plot(base.subsidy, base.effective_price, color="#c0504d", ls="--",
              label="Effective payment")
ax[0, 0].set_ylabel("CNY per enterprise")
ax[0, 0].legend(frameon=False, loc="upper left", fontsize=8.5)

ax[0, 1].plot(base.subsidy, base.participation * 100, color="#1f4e79", label="With subsidy")
ax[0, 1].plot(base.subsidy, base.withdrawal * 100, color="#c0504d", ls="--",
              label="After withdrawal")
ax[0, 1].set_ylabel("Participation (\%)")
ax[0, 1].legend(frameon=False, loc="upper left", fontsize=8.5)

ax[1, 0].plot(base.subsidy, base.spending / 1e6, color="#1f4e79")
ax[1, 0].axhline(c0.budget / 1e6, color="#7f7f7f", ls=":", label="Budget")
ax[1, 0].set_ylabel("Expected expenditure\n(million CNY)")
ax[1, 0].legend(frameon=False, loc="lower right", fontsize=8.5)

ax[1, 1].plot(base.subsidy, np.where(good, base.government, np.nan),
              color="#1f4e79", label="Government score")
ax[1, 1].plot(base.subsidy, np.where(good, base.exceed_risk, np.nan) * 100, "--",
              color="#c0504d", label="Overspend risk (\%)")
ax[1, 1].set_ylabel("Score / risk (\%)")
ax[1, 1].legend(frameon=False, loc="upper left", fontsize=8.5)

# Panel letters inside the plotting area, top-left, so no label floats above a panel.
letters = ["(a)", "(b)", "(c)", "(d)"]
for a, lab in zip(ax.flat, letters):
    a.text(0.02, 0.96, lab, transform=a.transAxes, ha="left", va="top",
           fontsize=10, fontweight="bold", color="#333333")
    a.set_xlabel("Subsidy (CNY per adopter)")
    a.grid(axis="y", color="#e6e6e6", lw=0.6)
    a.set_axisbelow(True)
fig.tight_layout(h_pad=1.4)
save(fig, "exp04_subsidy")

# ---- exp05: withdrawal ------------------------------------------------------
best = base[base.budget_feasible].sort_values(["government", "subsidy"],
                                              ascending=[False, True]).iloc[0]
no = pd.read_csv(TAB / "policy_benchmarks_rev.csv")
no = no[no.scenario == "no_subsidy"].iloc[0]
outcomes = pd.DataFrame([
    ("Subsidized", best.participation, best.profit, best.price),
    ("Withdrawal", best.withdrawal, best.withdrawal_profit, best.price),
    ("Re-optimized s=0", no.participation, no.profit, no.price)],
    columns=["scenario", "participation", "profit", "price"])
outcomes.to_csv(TAB / "withdrawal_rev.csv", index=False, encoding="utf-8-sig")
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
for a, col, fmt in zip(ax, ["participation", "profit", "price"],
                       ["{:.1%}", "{:.2f}", "{:.0f}"]):
    vals = outcol = outcomes[col].to_numpy(float)
    scale = 1e6 if col == "profit" else 1
    a.bar(outcomes.scenario, vals / scale,
          color=["#1f4e79", "#7a9cbf", "#2f6f5e"])
    a.set_title({"participation": "Participation",
                 "profit": "Profit (million CNY)",
                 "price": "Price (CNY)"}[col])
    a.tick_params(axis="x", rotation=15)
ax[1].axhline(0, color="k", lw=.8)
save(fig, "exp05_withdrawal")

# ---- exp06: sensitivity -----------------------------------------------------
sens = pd.read_csv(TAB / "sensitivity_rev.csv")
specs = [("eta", r"Recognition rate $\eta$"),
         ("information", "Information value (CNY)"),
         ("mu", r"Coverage weight $\mu$"),
         ("sigma", r"Logit scale $1/\theta$ (CNY)"),
         ("variable", "Variable cost (CNY/adopter)"),
         ("budget", "Expected budget (CNY)"),
         ("lam", r"Benefit weight $\lambda$")]
fig, ax = plt.subplots(3, 3, figsize=(12, 9))
for a, (par, label) in zip(ax.flat, specs):
    sub = sens[sens.parameter == par]
    a.plot(sub.value, sub.participation * 100, "o-", label="Participation")
    a.plot(sub.value, sub.withdrawal * 100, "s--", label="After withdrawal")
    a.set(xlabel=label, ylim=(0, 80))
    a.legend(fontsize=6)
for a in ax.flat[7:]:
    a.axis("off")
save(fig, "exp06_sensitivity")

print("regenerated exp04, exp05, exp06")
