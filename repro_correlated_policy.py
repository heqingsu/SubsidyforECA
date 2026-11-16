"""Reproduce the correlated-policy recalibration reported in Section 5.6.

For each latent correlation gamma, scan the provider's best responses on the
250 CNY joint price-subsidy grid, keep those whose Gaussian-copula overspend
probability satisfies alpha = 0.10, and select the highest government score.

Reported in the text: gamma = 0.10 -> s = 9,000 CNY, p = 26,750 CNY,
participation 39.32%, risk 5.4%, i.e. 1.15 pp below the independence-selected
policy. Output: outputs/tables/correlated_policy.csv
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from scipy.special import expit

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rf", HERE / "revision_final.py")
rf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rf)
spec2 = importlib.util.spec_from_file_location("rf2", HERE / "revision_final2.py")
m = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(m)

base = rf.C()
D = rf.build(rf.E0, rf.phi0, rf.w0, base)
fr, best, _ = rf.solve(D, base, dp=250, ds=250)
ALPHA, BUDGET = 0.10, 1_000_000.0

rows = []
for gamma in (0.0, 0.10, 0.20):
    cand = []
    for r in fr.itertuples():
        if r.configuration == "no_service" or r.subsidy <= 0:
            continue
        b = np.asarray(D[r.configuration]["b"], float)
        P = expit((b - r.price + r.subsidy) / base.sigma)
        thr = int(np.floor(BUDGET / r.subsidy))
        risk = m.copula_exceed(P, thr, gamma)
        if risk <= ALPHA:
            cand.append(dict(gamma=gamma, subsidy=float(r.subsidy), price=float(r.price),
                             configuration=r.configuration,
                             participation=float(P.mean()), risk=float(risk),
                             government=float(r.government), profit=float(r.profit)))
    cand.sort(key=lambda x: -x["government"])
    rows.extend(cand[:3])
    if cand:
        top = cand[0]
        print(f"gamma={gamma:.2f}  s={top['subsidy']:.0f} p={top['price']:.0f} "
              f"part={100*top['participation']:.2f}% risk={100*top['risk']:.2f}% "
              f"gov={top['government']:.5f} feasible={len(cand)}")

out = HERE / "outputs" / "tables" / "correlated_policy.csv"
pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
print("written", out)
