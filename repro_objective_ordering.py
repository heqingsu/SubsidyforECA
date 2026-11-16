"""Verify the claim in Section 3.5 of the revised manuscript:

    "across the four recognition regimes in Table (regimes), the score and
     the adjusted benefit index ... order the policies identically."

Re-runs the four recognition regimes of revision_final2.py at the same
250 CNY grid, records the government score J_G (lambda = 0.5, omega = 0)
and the adjusted benefit index I_adj for each selected policy, and checks
that both rankings coincide.

Output: outputs/tables/objective_ordering.csv
"""

import importlib.util
import sys
from dataclasses import replace

import pandas as pd

spec = importlib.util.spec_from_file_location("rf2", "revision_final2.py")
rf2 = importlib.util.module_from_spec(spec)
sys.modules["rf2"] = rf2
spec.loader.exec_module(rf2)

REGIMES = [
    ("No monetary recognition", replace(rf2.C(), eta=0.0, information=0.0)),
    ("Information value only", replace(rf2.C(), eta=0.0, information=5000.0)),
    ("Institutional recognition (baseline)", rf2.C()),
    ("Partial recognition (eta=0.4)", replace(rf2.C(), eta=0.4)),
]


def main():
    base = rf2.C()
    rows = []
    for label, cc in REGIMES:
        DD = rf2.build(rf2.E0, rf2.phi0, rf2.w0, cc)
        _, bb, _ = rf2.solve(DD, cc, dp=250, ds=250)
        _, nb = rf2.net_indicator(bb, DD, cc)
        rows.append(dict(regime=label,
                         government=bb["government"],
                         adjusted_index_million_cny=nb / 1e6,
                         subsidy=bb["subsidy"], price=bb["price"],
                         lam=cc.lam, omega=cc.omega))
        print(f"{label:38s} J_G={bb['government']:.4f}  I_adj={nb/1e6:+.3f} M CNY")
    d = pd.DataFrame(rows)
    order_gov = d.sort_values("government", ascending=False)["regime"].tolist()
    order_net = d.sort_values("adjusted_index_million_cny", ascending=False)["regime"].tolist()
    identical = order_gov == order_net
    print("\norder by score :", " > ".join(order_gov))
    print("order by index :", " > ".join(order_net))
    print("rankings identical:", identical)
    d["rankings_identical"] = identical
    d.to_csv(rf2.TAB / "objective_ordering.csv", index=False, encoding="utf-8-sig")
    print("written:", rf2.TAB / "objective_ordering.csv")
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
