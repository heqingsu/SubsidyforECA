"""Reproduce the construction diagnostics in Appendix Table A2 (fit tests).

Reads the distributed panel and tests each drawn attribute against the
distribution it was drawn from. Output: outputs/tables/fit_tests.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / "data" / "enterprises.csv", encoding="utf-8-sig")

x = np.log(d.electricity_MWh)
tests = [
    ("Annual electricity (log scale)", "Normal", x, stats.norm, (x.mean(), x.std(ddof=1))),
    ("Eligibility fraction", "U(0.40,0.90)", d.eligible_fraction.values, stats.uniform, (0.40, 0.50)),
    ("Unit value", "U(40,80)", d.unit_value.values, stats.uniform, (40, 40)),
    ("Implementation cost", "U(2,000,6,000)", d.extra_cost.values, stats.uniform, (2000, 4000)),
]

rows = []
for name, law, v, rv, args in tests:
    r = stats.kstest(v, rv.cdf, args=args)
    rows.append({"input": name, "law": law, "D": round(float(r.statistic), 4), "p": round(float(r.pvalue), 3)})
    print(f"{name:32s} {law:16s} D={r.statistic:.4f} p={r.pvalue:.3f}")

out = HERE / "outputs" / "tables" / "fit_tests.csv"
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
print("written", out)
