"""Re-run the legacy sensitivity and policy-benchmark specifications under the
revised baseline (subsidy cap 20,000 CNY, omega = 0, 500 CNY grids) so that every
number quoted in the revised manuscript is internally consistent.
"""
from pathlib import Path
from dataclasses import replace
import pandas as pd

import revision_experiments as R

TAB = R.TAB
c0 = R.Settings()
data, _ = R.prepare(R.E0, c0)

# ---- policy benchmarks -------------------------------------------------------
bench = []
for label, allowed, sm in [("no_subsidy", None, 0),
                           ("fixed_coarse", ["t24h_g1"], 20000),
                           ("fixed_fine", ["t1h_g8"], 20000),
                           ("endogenous", None, 20000)]:
    fr, be, pe = R.solve(data, replace(c0, smax=sm), 250, 250, allowed=allowed)
    bench.append(dict(scenario=label, **be))
    print(label, {k: be[k] for k in ("subsidy", "price", "participation", "profit", "spending")}, flush=True)
pd.DataFrame(bench).to_csv(TAB / "policy_benchmarks_rev.csv", index=False, encoding="utf-8-sig")

# ---- capped-at-10,000 comparison row -----------------------------------------
fr, be, pe = R.solve(data, replace(c0, smax=10000), 250, 250)
row = dict(scenario="capped_10000", **be)
pd.DataFrame([row]).to_csv(TAB / "capped10000.csv", index=False, encoding="utf-8-sig")
print("capped_10000", {k: be[k] for k in ("subsidy", "price", "participation", "profit", "spending")}, flush=True)

# ---- legacy sensitivity specs under the revised baseline ----------------------
specs = [("eta", [.4, .6, .8, 1]),
         ("information", [0, 2000, 5000]),
         ("mu", [0, .3, .6, .9, 1]),
         ("sigma", [5000, 10000, 20000]),
         ("variable", [0, 500, 1500]),
         ("budget", [250000, 500000, 1000000, 2000000]),
         ("lam", [0, .5, 1])]
rows = []
for par, vals in specs:
    for val in vals:
        cs = replace(c0, **{par: val})
        dd, _ = R.prepare(R.E0, cs)
        _, be, _ = R.solve(dd, cs, 500, 500)
        rows.append(dict(parameter=par, value=val, **be))
    print("sensitivity", par, flush=True)
sens = pd.DataFrame(rows)
sens.to_csv(TAB / "sensitivity_rev.csv", index=False, encoding="utf-8-sig")
cols = ["parameter", "value", "configuration", "subsidy", "price",
        "participation", "profit", "spending", "withdrawal"]
print(sens[cols].to_string(), flush=True)
