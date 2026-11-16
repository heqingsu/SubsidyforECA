"""Round-2 revision analysis.

Outputs (outputs/tables/):
  r2_eta_sweep.csv          recognition sweep 0.60-1.00: optimal policy, score, participation
  r2_resolution_scopes.csv  temporal-only vs spatial-only vs full configuration sets
  r2_welfare_decomposition.csv  welfare components and the adoption margin
  r2_breakeven_counterfactual.csv  fixed-configuration counterfactual at the break-even price
  r2_externality_breakeven.csv  break-even information externality per additional adopter
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rf = load(HERE / "revision_final.py", "rf")
base = rf.C()
D = rf.build(rf.E0, rf.phi0, rf.w0, base)
NAMES = list(D)


def solve_for(c, allowed=None, dp=250.0, ds=250.0, data=None):
    data = D if data is None else data
    if allowed is not None:
        data = {k: data[k] for k in allowed}
    fr, best, _ = rf.solve(data, c, dp=dp, ds=ds)
    zero = fr[fr.subsidy == 0].iloc[0]
    return fr, best, zero


# ----------------------------------------------------------------------
# 1. Recognition sweep eta = 0.60 ... 1.00 (reviewer 3, point 5c)
# ----------------------------------------------------------------------
rows = []
for eta in np.round(np.arange(0.00, 1.001, 0.05), 2):
    c = rf.C(eta=float(eta))
    DD = rf.build(rf.E0, rf.phi0, rf.w0, c)
    fr, best, zero = solve_for(c, data=DD)
    rows.append(dict(
        eta=float(eta),
        configuration=best["configuration"],
        subsidy=float(best["subsidy"]),
        price=float(best["price"]),
        participation=float(best["participation"]),
        score=float(best["government"]),
        score_zero=float(zero["government"]),
        participation_zero=float(zero["participation"]),
        score_gain=float(best["government"] - zero["government"]),
        spending=float(best["spending"]),
    ))
    print(f"eta={eta:.2f}  k={best['configuration']:8s} s={best['subsidy']:6.0f} "
          f"p={best['price']:6.0f} part={100*best['participation']:5.2f}% "
          f"score={best['government']:.4f} gain={best['government']-zero['government']:+.4f}")
eta_df = pd.DataFrame(rows)
eta_df.to_csv(OUT / "r2_eta_sweep.csv", index=False, encoding="utf-8-sig")

# ----------------------------------------------------------------------
# 2. Spatial dimension: temporal-only vs spatial-only vs full set
# ----------------------------------------------------------------------
scopes = {
    "Temporal only (1 zone)": ["t24h_g1", "t6h_g1", "t1h_g1"],
    "Spatial only (24 h)": ["t24h_g1", "t24h_g4", "t24h_g8"],
    "Full set (3 x 3)": NAMES,
    "Hourly only (3 zones)": ["t1h_g1", "t1h_g4", "t1h_g8"],
}
srows = []
for label, allowed in scopes.items():
    fr, best, zero = solve_for(base, allowed=allowed)
    d = D[best["configuration"]]
    srows.append(dict(
        scope=label,
        configurations=len(allowed),
        selected=best["configuration"],
        subsidy=float(best["subsidy"]),
        price=float(best["price"]),
        participation=float(best["participation"]),
        score=float(best["government"]),
        score_zero=float(zero["government"]),
        participation_zero=float(zero["participation"]),
        spending=float(best["spending"]),
        participation_gain=float(best["participation"] - zero["participation"]),
    ))
    print(f"{label:24s} k={best['configuration']:8s} s={best['subsidy']:6.0f} "
          f"part={100*best['participation']:5.2f}%  score={best['government']:.4f}")
pd.DataFrame(srows).to_csv(OUT / "r2_resolution_scopes.csv", index=False, encoding="utf-8-sig")

# ----------------------------------------------------------------------
# 3. Welfare decomposition and the markup / externality question
# ----------------------------------------------------------------------
def wparts(k, p, s, c=base):
    d = D[k]
    b = d["b"]
    V = b - p + s
    P = expit(V / c.sigma)
    A = P.sum()
    cs = c.sigma * np.logaddexp(0.0, V / c.sigma).sum()
    prof = (p - c.variable) * A - d["F"]
    return dict(A=A, participation=A / len(b), cs=float(cs),
                profit=float(prof), spending=float(s * A), b=b, P=P)


fr, best, zero = solve_for(base)
w0 = wparts(zero["configuration"], float(zero["price"]), 0.0)
w1 = wparts(best["configuration"], float(best["price"]), float(best["subsidy"]))

# marginal adopters: probability-weighted difference in adoption
dP = w1["P"] - w0["P"]
b_new = float((dP * w1["b"]).sum() / dP.sum()) if dP.sum() > 0 else np.nan
dA = float(w1["A"] - w0["A"])

# Fixed-configuration counterfactual at the break-even (zero economic profit) price.
# The price solves (p - v) * A(p; s) - F_k = 0 with the configuration held fixed at the
# one each regime selects. This is NOT a competitive equilibrium and not a welfare bound:
# it removes the provider's margin at a given configuration, nothing else.
def breakeven_price(k, s, c=base):
    d = D[k]
    b, F = d["b"], d["F"]

    def excess(p):
        a = expit((b - p + s) / c.sigma).sum()
        return (p - c.variable) * a - F

    lo, hi = float(c.variable), 60_000.0
    if excess(hi) < 0:
        return np.nan
    return brentq(excess, lo, hi, xtol=1e-6)

p0_be = breakeven_price(zero["configuration"], 0.0)
p1_be = breakeven_price(best["configuration"], float(best["subsidy"]))
w0b = wparts(zero["configuration"], p0_be, 0.0)
w1b = wparts(best["configuration"], p1_be, float(best["subsidy"]))
print(f"\nbreak-even prices: no support {p0_be:,.0f} (k={zero['configuration']}), "
      f"baseline {p1_be:,.0f} (k={best['configuration']})")

brows = []
for label, k, p_be, s_, w in [
        ("No support, break-even price", zero["configuration"], p0_be, 0.0, w0b),
        ("Baseline, break-even price", best["configuration"], p1_be,
         float(best["subsidy"]), w1b)]:
    for m in (1.0, 1.3):
        brows.append(dict(scenario=label, configuration=k, subsidy=s_, price=p_be,
                          participation=w["participation"], adopters=w["A"],
                          cs=w["cs"], profit=w["profit"], spending=w["spending"],
                          welfare=w["cs"] + w["profit"] - m * w["spending"], mcpf=m))
pd.DataFrame(brows).to_csv(OUT / "r2_breakeven_counterfactual.csv", index=False,
                           encoding="utf-8-sig")

rows = []
for label, w in [("Zero-subsidy market", w0), ("Baseline", w1)]:
    for m in (1.0, 1.3):
        rows.append(dict(scenario=label, mcpf=m,
                         participation=w["participation"], adopters=w["A"],
                         cs=w["cs"], profit=w["profit"], spending=w["spending"],
                         welfare=w["cs"] + w["profit"] - m * w["spending"]))
wd = pd.DataFrame(rows)
wd["dA_vs_zero"] = wd["adopters"] - w0["A"]
wd["marginal_adopter_value"] = b_new
wd.to_csv(OUT / "r2_welfare_decomposition.csv", index=False, encoding="utf-8-sig")

gain_gross = (w1["cs"] + w1["profit"]) - (w0["cs"] + w0["profit"])
print(f"\ngross gain CS+profit = {gain_gross:,.0f}; spending = {w1['spending']:,.0f}; "
      f"m* = {gain_gross / w1['spending']:.3f}")
print(f"extra adopters = {dA:.2f}; mean B of marginal adopters = {b_new:,.0f}")

# break-even externality per additional adopter: e* s.t. gain + e*dA = m*S
erows = []
for m in (1.0, 1.2, 1.3, 1.4, 1.5):
    need = m * w1["spending"] - gain_gross
    erows.append(dict(mcpf=m, net_gain=gain_gross - m * w1["spending"],
                      extra_adopters=dA, breakeven_externality=need / dA if dA > 0 else np.nan))
    print(f"m={m:.1f}: net gain {gain_gross - m*w1['spending']:+,.0f}; "
          f"e* = {need/dA:+,.0f} CNY per extra adopter")
pd.DataFrame(erows).to_csv(OUT / "r2_externality_breakeven.csv", index=False, encoding="utf-8-sig")

# break-even comparison: does the subsidy still raise welfare once the margin is removed?
for m in (1.0, 1.3):
    print(f"break-even price, m={m:.1f}: no support {w0b['cs']+w0b['profit']-m*w0b['spending']:,.0f} "
          f"vs baseline {w1b['cs']+w1b['profit']-m*w1b['spending']:,.0f} "
          f"(difference {w1b['cs']+w1b['profit']-m*w1b['spending']-(w0b['cs']+w0b['profit']-m*w0b['spending']):+,.0f})")
print("\nwritten:", sorted(p.name for p in OUT.glob("r2_*.csv")))
