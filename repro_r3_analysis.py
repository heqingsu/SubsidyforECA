"""R3 revision analysis: continuous provider pricing, welfare comparison,
lambda/normalisation sweeps, quadrature convergence, behavioural sensitivity.

Outputs (outputs/tables/):
  r3_continuous_price.csv    provider best-response price p*(s) around the baseline
  r3_welfare.csv             welfare decomposition of four policy scenarios
  r3_lambda_sweep.csv        government score weights lambda sweep
  r3_normalisation_sweep.csv href (and xref at omega>0) sweep
  r3_quadrature.csv          Gauss-Hermite node convergence of the copula risk
  r3_sigma_sweep.csv         logit scale sigma sweep (behavioural sensitivity)
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("rf", HERE / "revision_final.py")
rf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rf)
spec2 = importlib.util.spec_from_file_location("rf2", HERE / "revision_final2.py")
m = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(m)

base = rf.C()
D = rf.build(rf.E0, rf.phi0, rf.w0, base)
names = list(D)


# ----------------------------------------------------------------------
# 1. Continuous provider best-response price p*(s)
# ----------------------------------------------------------------------
def br_cont(s, c=base, data=D):
    """Continuous-price provider best response, mirroring solve()'s
    incumbent logic (configuration order, profit>=0 guard, no_service fallback)."""
    incumbent = 0.0
    best = dict(configuration="no_service", price=0.0, obj=0.0, profit=0.0,
                participation=0.0, spending=0.0)
    for name in names:
        d = data[name]
        b = d["b"]; n = len(b); F = d["F"]

        def neg_obj(p):
            A = expit((b - p + s) / c.sigma).sum()
            profit = (p - c.variable) * A - F
            obj = c.mu * A / n + (1 - c.mu) * profit / c.pref
            return -obj, profit, A

        # coarse scan then bounded local refinement
        grid = np.linspace(0.0, c.pmax, 241)
        vals = np.array([neg_obj(p)[0] for p in grid])
        ok = np.array([neg_obj(p)[1] >= 0 for p in grid])
        if not ok.any():
            continue
        vals[~ok] = np.inf
        j = int(np.argmin(vals))
        lo = max(grid[max(j - 1, 0)], 1.0)
        hi = grid[min(j + 1, len(grid) - 1)]
        res = minimize_scalar(lambda p: neg_obj(p)[0], bounds=(lo, hi),
                              method="bounded",
                              options={"xatol": 0.05})
        p = float(res.x)
        v, profit, A = neg_obj(p)
        v = -v
        obj = v
        if profit < 0:
            continue
        if obj < incumbent - c.tau:
            continue
        incumbent = max(incumbent, obj)
        if obj >= incumbent - c.tau:
            best = dict(configuration=name, price=p, obj=float(obj),
                        profit=float(profit), participation=float(A / n),
                        spending=float(s * A))
    return best


print("== continuous provider best response ==")
rows = []
for s in np.arange(10000, 12501, 50):
    r = br_cont(float(s))
    rows.append(dict(subsidy=s, **r))
    print(f"s={s:6.0f}  k={r['configuration']:>16s}  p*={r['price']:8.2f}  "
          f"part={100*r['participation']:.2f}%  obj={r['obj']:.6f}")
cp = pd.DataFrame(rows)
cp.to_csv(OUT / "r3_continuous_price.csv", index=False, encoding="utf-8-sig")

# slope dp/ds at the baseline subsidy s0 = 11,250
s0 = 11250.0
w = np.polyfit(cp.subsidy, cp.price, 1)
slope = w[0]
# central difference on +/-250 window
p_lo = br_cont(s0 - 250)["price"]; p_hi = br_cont(s0 + 250)["price"]
cd = (p_hi - p_lo) / 500.0
print(f"\nOLS slope dp/ds = {slope:.4f}  ({100*slope:.1f}% incidence)")
print(f"central difference (+/-250) = {cd:.4f}  ({100*cd:.1f}%)")
# per-step incidence across the reported discrete steps
step = cp.subsidy.diff()
dstep = cp.price.diff() / step
print("per-50-CNY-step incidence range: %.1f%% .. %.1f%%"
      % (100 * dstep[1:].min(), 100 * dstep[1:].max()))

# continuous-price baseline offer at the selected subsidy
cb = br_cont(s0)
P = expit((D[cb["configuration"]]["b"] - cb["price"] + s0) / base.sigma)
spend = s0 * P.sum()
gov = base.lam * (P @ D[cb["configuration"]]["B"]) / base.href \
    + (1 - base.lam) * expit(D[cb["configuration"]]["b"] / base.sigma).mean()
print(f"\ncontinuous-price baseline offer at s={s0:.0f}: k={cb['configuration']} "
      f"p={cb['price']:.1f} part={100*P.mean():.2f}% spend={spend:,.0f} gov={gov:.4f}")


# ----------------------------------------------------------------------
# 2. Welfare comparison (log-sum consumer surplus + provider profit
#    - marginal cost of public funds * spending)
# ----------------------------------------------------------------------
def welfare(config, price, subsidy, c=base, data=D, rho=0.0, nodes=41):
    if config == "no_service":
        return dict(configuration="no_service", price=0.0, subsidy=0.0,
                    participation=0.0, cs=0.0, profit=0.0, spending=0.0,
                    risk_indep=0.0)
    d = data[config]
    b = d["b"]
    V = b - price + subsidy
    P = expit(V / c.sigma)
    A = P.sum()
    cs = c.sigma * np.logaddexp(0.0, V / c.sigma).sum()   # log-sum CS, CNY
    profit = (price - c.variable) * A - d["F"]
    spend = subsidy * A
    thr = int(np.floor(c.budget / subsidy)) if subsidy > 0 else 0
    risk = m.copula_exceed(P, thr, rho, nodes=nodes) if subsidy > 0 else 0.0
    return dict(configuration=config, price=price, subsidy=subsidy,
                participation=float(P.mean()), cs=float(cs),
                profit=float(profit), spending=float(spend),
                risk_indep=float(risk))

fr, best, _ = rf.solve(D, base, dp=250, ds=250)
r0 = fr[fr.subsidy == 0].iloc[0].to_dict()          # zero-subsidy market
r1 = fr[fr.subsidy == 11250].iloc[0].to_dict()      # baseline

scen = [("No service", "no_service", 0.0, 0.0),
        ("Zero-subsidy market", r0["configuration"], float(r0["price"]), 0.0),
        ("Baseline (expected budget)", r1["configuration"], float(r1["price"]), 11250.0),
        ("Chance constraint (alpha=0.10)", "t1h_g8", 27000.0, 10250.0),
        ("Baseline, continuous price", cb["configuration"], cb["price"], 11250.0)]
wrows = []
for label, k, p, s in scen:
    w_ = welfare(k, p, s)
    w_["scenario"] = label
    for mcpf in (1.0, 1.3):
        w_[f"W_mcpf_{mcpf:.1f}"] = w_["cs"] + w_["profit"] - mcpf * w_["spending"]
    wrows.append(w_)
    print(f"\n{label}: k={k} p={p:.0f} s={s:.0f}")
    print(f"  part={100*w_['participation']:.2f}%  CS={w_['cs']:,.0f}  "
          f"profit={w_['profit']:,.0f}  spend={w_['spending']:,.0f}")
    print(f"  W(m=1.0)={w_['W_mcpf_1.0']:,.0f}  W(m=1.3)={w_['W_mcpf_1.3']:,.0f}")
wf = pd.DataFrame(wrows)
# break-even marginal cost of public funds between zero-subsidy and baseline
gains = (wf[wf.scenario == "Baseline (expected budget)"].cs.iloc[0]
         + wf[wf.scenario == "Baseline (expected budget)"].profit.iloc[0]
         - wf[wf.scenario == "Zero-subsidy market"].cs.iloc[0]
         - wf[wf.scenario == "Zero-subsidy market"].profit.iloc[0])
spend_b = wf[wf.scenario == "Baseline (expected budget)"].spending.iloc[0]
m_star = gains / spend_b
print(f"break-even MCPF m* = {m_star:.3f} (subsidy welfare-improving iff m < m*)")
wf.to_csv(OUT / "r3_welfare.csv", index=False, encoding="utf-8-sig")


# ----------------------------------------------------------------------
# 3. Lambda sweep (government score weights)
# ----------------------------------------------------------------------
print("\n== lambda sweep ==")
lrows = []
for lam in (0.2, 0.35, 0.5, 0.65, 0.8):
    c = rf.C(lam=lam)
    DD = rf.build(rf.E0, rf.phi0, rf.w0, c)
    _, bb, _ = rf.solve(DD, c, dp=250, ds=250)
    lrows.append(dict(lam=lam, configuration=bb["configuration"],
                      subsidy=bb["subsidy"], price=bb["price"],
                      participation=bb["participation"], score=bb["government"]))
    print(f"lambda={lam:.2f}: k={bb['configuration']} s={bb['subsidy']:.0f} "
          f"p={bb['price']:.0f} part={100*bb['participation']:.2f}%")
pd.DataFrame(lrows).to_csv(OUT / "r3_lambda_sweep.csv", index=False, encoding="utf-8-sig")


# ----------------------------------------------------------------------
# 4. Normalisation sweep (href at omega=0; href & xref at omega=0.005)
# ----------------------------------------------------------------------
print("\n== normalisation sweep ==")
nrows = []
for href in (5e6, 1e7, 2e7):
    c = rf.C(href=href)
    DD = rf.build(rf.E0, rf.phi0, rf.w0, c)
    _, bb, _ = rf.solve(DD, c, dp=250, ds=250)
    nrows.append(dict(omega=0.0, href=href, xref=base.xref,
                      configuration=bb["configuration"], subsidy=bb["subsidy"],
                      price=bb["price"], participation=bb["participation"]))
    print(f"omega=0     href={href:.0e}: s={bb['subsidy']:.0f} p={bb['price']:.0f}")
for xref in (5e5, 1e6, 2e6):
    c = rf.C(omega=0.005, xref=xref)
    DD = rf.build(rf.E0, rf.phi0, rf.w0, c)
    _, bb, _ = rf.solve(DD, c, dp=250, ds=250)
    nrows.append(dict(omega=0.005, href=base.href, xref=xref,
                      configuration=bb["configuration"], subsidy=bb["subsidy"],
                      price=bb["price"], participation=bb["participation"]))
    print(f"omega=0.005 xref={xref:.0e}: s={bb['subsidy']:.0f} p={bb['price']:.0f}")
pd.DataFrame(nrows).to_csv(OUT / "r3_normalisation_sweep.csv", index=False,
                           encoding="utf-8-sig")


# ----------------------------------------------------------------------
# 5. Quadrature convergence of the copula overspend risk
# ----------------------------------------------------------------------
print("\n== quadrature convergence ==")
d = D[r1["configuration"]]
P = expit((d["b"] - r1["price"] + 11250.0) / base.sigma)
thr = int(np.floor(base.budget / 11250.0))
qrows = []
for nodes in (9, 15, 21, 41, 61, 101):
    risk = m.copula_exceed(P, thr, 0.10, nodes=nodes)
    qrows.append(dict(nodes=nodes, risk=risk))
    print(f"nodes={nodes:3d}: risk={100*risk:.4f}%")
pd.DataFrame(qrows).to_csv(OUT / "r3_quadrature.csv", index=False, encoding="utf-8-sig")


# ----------------------------------------------------------------------
# 6. Behavioural sensitivity: logit scale sigma and recognition multiplier eta
# ----------------------------------------------------------------------
print("\n== sigma / eta sweep ==")
brows = []
for sig in (5000.0, 7500.0, 10000.0, 15000.0, 20000.0):
    c = rf.C(sigma=sig)
    DD = rf.build(rf.E0, rf.phi0, rf.w0, c)
    _, bb, _ = rf.solve(DD, c, dp=250, ds=250)
    brows.append(dict(param="sigma", value=sig, configuration=bb["configuration"],
                      subsidy=bb["subsidy"], price=bb["price"],
                      participation=bb["participation"]))
    print(f"sigma={sig:6.0f}: k={bb['configuration']} s={bb['subsidy']:.0f} "
          f"p={bb['price']:.0f} part={100*bb['participation']:.2f}%")
for eta in (0.6, 0.8, 1.0):
    c = rf.C(eta=eta)
    DD = rf.build(rf.E0, rf.phi0, rf.w0, c)
    _, bb, _ = rf.solve(DD, c, dp=250, ds=250)
    brows.append(dict(param="eta", value=eta, configuration=bb["configuration"],
                      subsidy=bb["subsidy"], price=bb["price"],
                      participation=bb["participation"]))
    print(f"eta={eta:.2f}  : k={bb['configuration']} s={bb['subsidy']:.0f} "
          f"p={bb['price']:.0f} part={100*bb['participation']:.2f}%")
pd.DataFrame(brows).to_csv(OUT / "r3_sigma_sweep.csv", index=False, encoding="utf-8-sig")

print("\nAll R3 outputs written to", OUT)
