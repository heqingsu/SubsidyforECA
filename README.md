# SubsidyforECA — Replication package

Replication package for the manuscript **"Subsidy Design for Electricity Carbon
Accounting Services: Resolution Choice, Price Repricing, and Fiscal Risk"**

The package contains the calibrated working inputs, the solving and plotting
scripts, and the output tables of every reported result. The enterprise
profiles are generated from fitted distributions and contain no tenant
identifiers; the underlying park records are covered by a confidentiality
agreement and are not part of this repository.

## Environment

- Python 3.13.14, NumPy 2.5.4, SciPy 1.18.1, pandas 3.0.5, Matplotlib 3.11.2
- A single laptop; the policy comparisons of Section 5 complete in about 40 s

```bash
pip install -r requirements.txt
python revision_final.py        # baseline policy comparisons
python revision_final2.py       # recognition regimes, instruments, chance grid
```

## Repository layout

```
*.py        solving, sensitivity, figure and reproduction scripts
data/       calibrated enterprise profiles (200 rows) and the 24x8 factor table
outputs/    tables (*.csv, *.json) and figures (*.pdf) as used in the paper
```

## Map from manuscript items to scripts

| Manuscript item | Script | Output |
|---|---|---|
| Table 3 (recognition regimes), Table 4 (objectives) | `revision_final2.py` | `outputs/tables/final2_regimes.csv`, `final2_objectives.csv` |
| Table 5 (welfare), continuous price, λ/normalisation sweeps, quadrature ladder | `repro_r3_analysis.py` | `outputs/tables/r3_*.csv` |
| Table 6 (verified policy results) | `revision_final.py` | `outputs/tables/final_summary.json` |
| Tables 8–9 (band, grid convergence) | `revision_final.py` | `outputs/tables/final_summary.json` |
| Chance-constraint grid | `revision_final2.py` | `outputs/tables/final2_chance_grid.csv` |
| Correlated-recalibration table | `repro_correlated_policy.py` | `outputs/tables/final_common_shock.csv` |
| Instruments table | `revision_final2.py` | `outputs/tables/final2_instruments.csv` |
| Recognition sweep (Table, η = 0–1) | `repro_round2.py` | `outputs/tables/r2_eta_sweep.csv` |
| Resolution-scope comparison (temporal-only) | `repro_round2.py` | `outputs/tables/r2_resolution_scopes.csv` |
| Welfare decomposition, externality break-even | `repro_round2.py` | `outputs/tables/r2_welfare_decomposition.csv`, `r2_externality_breakeven.csv` |
| Break-even-price counterfactual (fixed configuration, m = 1 and 1.3) | `repro_round2.py` | `outputs/tables/r2_breakeven_counterfactual.csv` |
| Common-shock dependence check | `revision_final2.py` | `outputs/tables/final_common_shock.csv` | |
| Input provenance and fit diagnostics | `repro_fit_tests.py` | `outputs/tables/fit_tests.csv` |
| Government-objective ranking claim | `repro_objective_ordering.py` | `outputs/tables/objective_ordering.csv` |
| Figure 4 (attribution capability / cost of measurement) | `fig4_attributed_change.py` | `outputs/figures/rev04_attributed_change.pdf` |
| Figure 6 (provider responses, 2x2) | `fig6_subsidy_response.py` | `outputs/figures/exp04_subsidy.pdf` |
| Figures 3 and 10 | `revision_figures_v2.py` | `outputs/figures/exp02_aggregation.pdf`, `final_resolution_map.pdf` |
| Figure 4 | `revision_figures.py` | `outputs/figures/exp04_subsidy.pdf` |
| Figures 5–10 (settings based) | `revision_figures.py` | `outputs/figures/exp0*.pdf`, `rev0*.pdf` |

## Checks

The archive reports checks of energy conservation, benchmark identities, price
derivatives, demand invariance and expenditure accounting. All numerical tables
in the manuscript were regenerated from this package in the recorded
environment, and the reported values reproduce within rounding.

## Data dictionary

`data/enterprises.csv`: one row per enterprise — `id`, annual electricity
`electricity_MWh`, delivery `region` (0–7), `shift` category,
`eligible_fraction` (reschedulable share), `unit_value` (CNY per tonne of
recognised difference) and `extra_cost` (implementation cost, CNY).

`data/hourly_factors.csv`: hourly emission factors by zone — `hour`,
`region`, `factor` (kgCO2e/kWh) and `reference_weight` (load weights used for
aggregation).

## License

Released under the MIT License. The calibrated inputs are desensitized
derivatives of park operating statistics; they are released for replication of
the manuscript's results only.

