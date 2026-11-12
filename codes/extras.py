"""create by hqs
"""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
import pandas as pd
from scipy.special import expit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import revision_experiments as R

OUT, FIG, TAB = R.OUT, R.FIG, R.TAB
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})

c0 = R.Settings()
data, agg = R.prepare(R.E0, c0)

sweep = pd.read_csv(TAB / "omega_sweep.csv")
cc = pd.read_csv(TAB / "chance_constrained.csv")
boot = pd.read_csv(TAB / "bootstrap_1000.csv")

# ---- extended resolution map ------------------------------------------------
cost_scales = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 50.0]
etas = [0.2, 0.4, 0.6, 0.8, 1.0]
rows = []
for cs in cost_scales:
    for et in etas:
        ck = replace(c0, cost_scale=cs, eta=et)
        dd, _ = R.prepare(R.E0, ck)
        _, be, _ = R.solve(dd, ck, 500, 500)
        rows.append(dict(cost_scale=cs, eta=et, configuration=be["configuration"],
                         subsidy=be["subsidy"], price=be["price"],
                         participation=be["participation"],
                         government=be["government"], spending=be["spending"]))
    print("map", cs, flush=True)
rmap = pd.DataFrame(rows)
rmap.to_csv(TAB / "resolution_map_extended.csv", index=False, encoding="utf-8-sig")
print(rmap.pivot(index="eta", columns="cost_scale", values="configuration").to_string(), flush=True)

# ---- recognised reduction ---------------------------------------------------
rec = pd.read_csv(TAB / "recognised_reduction.csv")
rec["cost_per_tonne"] = np.where(rec.recognised_tCO2 > 1.0,
                                 rec.fixed_cost / rec.recognised_tCO2, np.nan)
rec.to_csv(TAB / "recognised_reduction.csv", index=False, encoding="utf-8-sig")

# ---- figures ----------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
ax[0].plot(sweep.omega, sweep.subsidy / 1000, "o-", color="#1f4e79")
ax[0].axhline(10, ls=":", color="grey")
ax[0].annotate("original cap", (sweep.omega.max() * .55, 10.8), fontsize=7, color="grey")
ax[0].set(xlabel=r"Fiscal weight $\omega$", ylabel="Selected subsidy (thousand CNY)",
          title="(a) Subsidy schedule")
ax[1].plot(sweep.omega, sweep.participation * 100, "o-", label="With subsidy")
ax[1].plot(sweep.omega, sweep.withdrawal * 100, "s--", label="After withdrawal")
ax[1].set(xlabel=r"Fiscal weight $\omega$", ylabel="Percent", title="(b) Adoption")
ax[1].legend(fontsize=7)
ax[2].plot(sweep.omega, sweep.spending / 1e6, "o-", label="Expected expenditure")
ax[2].plot(sweep.omega, sweep.exceed_risk * 100 / 100, "^-", label=r"Overspend risk ($\times$100)")
ax[2].axhline(c0.budget / 1e6, ls=":", color="grey")
ax[2].set(xlabel=r"Fiscal weight $\omega$", ylabel="Million CNY", title="(c) Expenditure")
ax[2].legend(fontsize=6)
fig.tight_layout()
fig.savefig(FIG / "rev01_omega_sweep.png", dpi=200, bbox_inches="tight")
fig.savefig(FIG / "rev01_omega_sweep.pdf", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ok = cc.dropna(subset=["subsidy"])
ax[0].plot(ok.alpha * 100, ok.participation * 100, "o-", color="#1f4e79")
ax[0].invert_xaxis()
ax[0].set(xlabel="Risk tolerance (%)", ylabel="Participation (%)",
          title="(a) Participation cost of risk control")
for _, r in ok.iterrows():
    ax[0].annotate(f"{r.subsidy/1000:g}k", (r.alpha * 100, r.participation * 100),
                   textcoords="offset points", xytext=(0, 6), fontsize=6, ha="center")
ax[1].plot(ok.alpha * 100, ok.spending / 1e6, "o-", label="Expected expenditure")
ax2 = ax[1].twinx()
ax2.plot(ok.alpha * 100, ok.exceed_risk * 100, "s--", color="#a4552f", label="Overspend risk")
ax2.set_ylabel("Overspend probability (%)")
ax2.spines["top"].set_visible(False)
ax[1].invert_xaxis()
ax[1].set(xlabel="Risk tolerance (%)", ylabel="Million CNY", title="(b) Instruments and risk")
h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax[1].legend(h1 + h2, l1 + l2, fontsize=6, loc="center left")
fig.tight_layout()
fig.savefig(FIG / "rev02_chance_constrained.png", dpi=200, bbox_inches="tight")
fig.savefig(FIG / "rev02_chance_constrained.pdf", bbox_inches="tight")
plt.close(fig)

order = ["t24h_g1", "t24h_g4", "t24h_g8", "t6h_g1", "t6h_g4", "t6h_g8",
         "t1h_g1", "t1h_g4", "t1h_g8"]
codes = {n: i for i, n in enumerate(order)}; codes["no_service"] = len(order)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
grid = rmap.pivot(index="eta", columns="cost_scale", values="configuration")
grid = grid.apply(lambda col: col.map(codes)).to_numpy(float)
im = ax.imshow(grid, aspect="auto", cmap="viridis", origin="lower",
               vmin=0, vmax=len(order))
ax.set_xticks(range(len(cost_scales)), [f"{x:g}" for x in cost_scales])
ax.set_yticks(range(len(etas)), [f"{x:g}" for x in etas])
ax.set(xlabel="Cost multiplier on configuration cost", ylabel=r"Monetary recognition rate $\eta$",
       title="Selected configuration")
for i in range(len(etas)):
    for j in range(len(cost_scales)):
        lab = "none" if grid[i, j] == len(order) else order[int(grid[i, j])]
        ax.text(j, i, lab, ha="center", va="center", fontsize=6, color="w")
fig.tight_layout()
fig.savefig(FIG / "rev03_resolution_map.png", dpi=200, bbox_inches="tight")
fig.savefig(FIG / "rev03_resolution_map.pdf", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
o = rec.set_index("configuration").loc[order].reset_index()
ax[0].barh(o.configuration, o.recognised_tCO2 / 1000, color="#2f6f5e")
ax[0].set(xlabel="Recognised reduction (thousand tCO$_2$/year)", title="(a) Attribution capability")
cp = o.cost_per_tonne.fillna(0)
bars = ax[1].barh(o.configuration, cp, color="#a4552f")
ax[1].set(xlabel="Fixed cost per recognised tonne (CNY/tCO$_2$)", title="(b) Unit cost")
for i, v in enumerate(o.cost_per_tonne):
    if np.isnan(v):
        ax[1].text(0.02, i, "no detectable variation", va="center", fontsize=6)
fig.tight_layout()
fig.savefig(FIG / "rev04_recognised_reduction.png", dpi=200, bbox_inches="tight")
fig.savefig(FIG / "rev04_recognised_reduction.pdf", bbox_inches="tight")
plt.close(fig)

# ---- summary ----------------------------------------------------------------
summ_path = OUT / "summary.json"
summary = json.loads(summ_path.read_text(encoding="utf8")) if summ_path.exists() else {}
base = pd.read_csv(TAB / "subsidy_response.csv")
best = base[base.budget_feasible].sort_values(["government","subsidy"],ascending=[False,True]).iloc[0].to_dict()
summary.setdefault("settings", {})
summary["baseline"] = {k: (float(v) if isinstance(v,(int,float,np.floating)) else v) for k,v in best.items()}
summary["resolution_map"] = {"config_by_cost": {str(k): v for k, v in
                                                rmap.groupby("cost_scale").configuration.first().items()},
                             "switching_cost_multiplier": float(
                                 rmap[rmap.configuration != "t1h_g8"].cost_scale.min())
                             if (rmap.configuration != "t1h_g8").any() else None}
summ_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                default=lambda v: v.item() if hasattr(v,"item") else str(v)), encoding="utf8")
print(json.dumps(summary["resolution_map"], ensure_ascii=False, indent=1))
print("FIGURE SET WRITTEN")
