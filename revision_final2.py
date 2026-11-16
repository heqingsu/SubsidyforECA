"""Second-round verification for the reviewer comments on the Energies submission.

Fixes / additions relative to revision_final.py
  (A) The net-benefit indicator is now computed for *every* selected policy,
      including zero-subsidy offers. revision_final.py guarded the implementation
      cost behind `subsidy > 0`, which forced the no-recognition regime to 0.000.
      The indicator is renamed: it is an enterprise-surplus indicator net of
      subsidy and implementation cost, not a welfare measure.
  (B) Correlated adoption is modelled two ways:
        - logit-scale common shock (`shock_exceed`, unchanged) -> a stress test
          that shifts the whole willingness distribution, so E[P_i(u)] != P_i;
        - Gaussian copula with equicorrelation rho (`copula_exceed`) -> preserves
          each marginal adoption probability exactly and moves only dependence.
  (C) The chance-constrained policy is re-solved on 500/250/100/50 CNY joint
      price-subsidy grids, so the "costs about one point of participation"
      statement is checked against grid resolution.
  (D) Instrument benchmarks disclose how the price-fixed subsidy is obtained
      (budget-boundary bisection on a 1 CNY grid) and report it alongside the
      250 CNY grid value.
"""
from pathlib import Path
from dataclasses import replace
import json, time
import numpy as np
import pandas as pd
from scipy.special import expit, logit, ndtr, ndtri
import importlib.util

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rf", HERE / "revision_final.py")
rf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rf)

E0, phi0, w0, C = rf.E0, rf.phi0, rf.w0, rf.C
build, solve, pbinom_exceed, shock_exceed = rf.build, rf.solve, rf.pbinom_exceed, rf.shock_exceed
OUT, TAB = rf.OUT, rf.TAB
EXTRA = E0.extra_cost.to_numpy()
ALPHA = .10


# ---------------------------------------------------------------- (A) indicator
def net_indicator(best, data, c):
    """Enterprise surplus net of subsidy paid out and implementation cost.

    NB = sum_i B_i P_i - s sum_i P_i - sum_i h_i P_i.
    The middle term is a transfer, not a cost, so this is NOT a welfare measure;
    it is the indicator the tables report.
    """
    if best["configuration"] == "no_service" or best["configuration"] not in data:
        return 0., 0.
    d = data[best["configuration"]]
    P = expit((d["b"] - best["price"] + best["subsidy"]) / c.sigma)
    impl = float(np.sum(P * EXTRA))
    nb = float(best["benefit"] - best["spending"] - impl)
    return impl, nb


# ------------------------------------------------------- (B) copula correlation
def copula_exceed(p, thr, rho, nodes=41):
    """P(sum of adoptions > thr) with a Gaussian copula, equicorrelation rho.

    U_i = Phi(sqrt(rho) z + sqrt(1-rho) e_i), adopt iff U_i <= P_i. Conditional on
    z the indicators are independent with q_i(z) = Phi((Phi^-1(P_i) - sqrt(rho) z)
    / sqrt(1-rho)), and E_z[q_i(z)] = P_i, so every marginal is preserved exactly.
    """
    p = np.clip(p, 1e-12, 1 - 1e-12)
    if rho <= 0:
        return pbinom_exceed(p, thr)
    x, wq = np.polynomial.hermite_e.hermegauss(nodes)
    wq = wq / wq.sum()
    z = np.sqrt(2.) * x                       # standard normal nodes
    L = ndtri(p)
    sr, si = np.sqrt(rho), np.sqrt(1. - rho)
    tot = 0.
    for zi, wi in zip(z, wq):
        q = ndtr((L - sr * zi) / si)
        tot += wi * pbinom_exceed(q, thr)
    return float(tot)


# ---------------------------------------------- (C) chance-constrained selection
def chance_policy(fr, data, c, alpha=ALPHA, rho=0., rf_exceed=None):
    """Best budget-feasible offer whose exceedance risk satisfies alpha."""
    ex = rf_exceed or pbinom_exceed
    fr = fr.copy()
    fr["exceed_risk"] = np.nan
    for k in fr.index:
        r = fr.loc[k]
        if not r.budget_feasible:
            continue
        if r.subsidy <= 0 or r.configuration == "no_service":
            fr.loc[k, "exceed_risk"] = 0.
            continue
        d = data[r.configuration]
        P = expit((d["b"] - r.price + r.subsidy) / c.sigma)
        fr.loc[k, "exceed_risk"] = ex(P, np.floor(c.budget / r.subsidy))
    ok = fr[(fr.exceed_risk <= alpha + 1e-12) & fr.budget_feasible]
    if not len(ok):
        return None, fr
    pick = ok.sort_values(["government", "subsidy"], ascending=[False, True]).iloc[0]
    return pick, fr


def main():
    t0 = time.perf_counter()
    base = C()
    D = build(E0, phi0, w0, base)
    S = {}

    # ---------------- (A) recognition regimes with a fully evaluated indicator
    rows = []
    for label, cc in [("No monetary recognition", replace(base, eta=0., information=0.)),
                      ("Information value only", replace(base, eta=0., information=5000.)),
                      ("Institutional recognition (baseline)", base),
                      ("Partial recognition (eta=0.4)", replace(base, eta=.4))]:
        DD = build(E0, phi0, w0, cc)
        _, bb, _ = solve(DD, cc, dp=250, ds=250)
        impl, nb = net_indicator(bb, DD, cc)
        rows.append(dict(regime=label, subsidy=bb["subsidy"], price=bb["price"],
                         configuration=bb["configuration"],
                         participation=bb["participation"], profit=bb["profit"],
                         spending=bb["spending"], benefit=bb["benefit"],
                         impl_cost=impl, net_indicator=nb, exceed=bb["exceed_risk"],
                         adopters=bb["participation"] * len(E0)))
        print(f"[A] {label:38s} s={bb['subsidy']:7.0f} p={bb['price']:7.0f} "
              f"part={bb['participation']:.4f} impl={impl/1e6:.3f} NB={nb/1e6:.3f}")
    reg = pd.DataFrame(rows)
    reg.to_csv(TAB / "final2_regimes.csv", index=False, encoding="utf-8-sig")
    S["regimes"] = reg.to_dict("records")

    # ---------------- (A) objective variants with the same indicator
    rows = []
    for label, cc in [("Expected budget, fixed-offer withdrawal", base),
                      ("No withdrawal term (lambda=1)", replace(base, lam=1.0))]:
        DD = build(E0, phi0, w0, cc)
        _, bb, _ = solve(DD, cc, dp=250, ds=250)
        impl, nb = net_indicator(bb, DD, cc)
        rows.append(dict(objective=label, subsidy=bb["subsidy"], price=bb["price"],
                         configuration=bb["configuration"],
                         participation=bb["participation"], spending=bb["spending"],
                         benefit=bb["benefit"], impl_cost=impl, net_indicator=nb,
                         exceed=bb["exceed_risk"]))
    # post-withdrawal re-optimised offer at s = 0
    fr250, best250, _ = solve(D, base, dp=250, ds=250)
    zero = fr250[fr250.subsidy == 0].iloc[0].to_dict()
    impl, nb = net_indicator(zero, D, base)
    rows.append(dict(objective="Re-optimised offer after withdrawal (s=0)",
                     subsidy=0., price=zero["price"], configuration=zero["configuration"],
                     participation=zero["participation"], spending=0.,
                     benefit=zero["benefit"], impl_cost=impl, net_indicator=nb, exceed=0.))
    obj = pd.DataFrame(rows)
    obj.to_csv(TAB / "final2_objectives.csv", index=False, encoding="utf-8-sig")
    print(obj[["objective", "subsidy", "participation", "impl_cost", "net_indicator"]].to_string(index=False))
    S["objectives"] = obj.to_dict("records")

    # ---------------- (B) correlation, marginals preserved vs shock
    dsel = D[best250["configuration"]]
    P = expit((dsel["b"] - best250["price"] + best250["subsidy"]) / base.sigma)
    thr = np.floor(base.budget / best250["subsidy"])
    print(f"[B] baseline: s={best250['subsidy']:.0f} p={best250['price']:.0f} "
          f"adopters={P.sum():.2f} thr={thr:.0f}")
    print(f"    mean adoption prob under shock su=1: "
          f"{(ndtr(ndtri(P) + 1.0)).mean():.4f} vs {P.mean():.4f}")
    rows = []
    for rho in (0., .05, .10, .20, .30, .50):
        rows.append(dict(rho=rho, risk=copula_exceed(P, thr, rho)))
    corr = pd.DataFrame(rows)
    corr.to_csv(TAB / "final2_correlation.csv", index=False, encoding="utf-8-sig")
    print(corr.to_string(index=False))
    S["correlation"] = corr.to_dict("records")
    # mean-preserving check: average adoption probability at rho and under the shock
    S["marginal_check"] = dict(
        mean_p=float(P.mean()),
        mean_p_shock_1=float(ndtr(ndtri(P) + 1.0).mean()),
        mean_p_shock_2=float(ndtr(ndtri(P) + 2.0).mean()),
        copula_mean_by_rho={str(r): float(np.mean([ndtr((ndtri(P) - np.sqrt(r) * z) / np.sqrt(1 - r))
                                                   for z in np.sqrt(2) * np.polynomial.hermite_e.hermegauss(41)[0]]))
                            for r in (0., .2, .5) if r < 1})

    # ---------------- (C) chance-constrained policy across grids
    rows = []
    for g in (500, 250, 100, 50):
        fr, best, info = solve(D, base, dp=g, ds=g)
        pick, _frr = chance_policy(fr, D, base)
        impl_c, nb_c = net_indicator(pick.to_dict(), D, base)
        impl_u, nb_u = net_indicator(best, D, base)
        rows.append(dict(grid=g,
                         s_uncon=best["subsidy"], p_uncon=best["price"],
                         part_uncon=best["participation"], risk_uncon=best["exceed_risk"],
                         s_cc=pick["subsidy"], p_cc=pick["price"],
                         part_cc=pick["participation"], risk_cc=pick["exceed_risk"],
                         cost_pp=100 * (best["participation"] - pick["participation"]),
                         gov_uncon=best["government"], gov_cc=pick["government"],
                         impl_cc=impl_c, nb_cc=nb_c, nb_uncon=nb_u,
                         config_cc=pick["configuration"]))
        print(f"[C] grid {g:4d}: uncon s={best['subsidy']:6.0f} risk={best['exceed_risk']:.4f} | "
              f"cc s={pick['subsidy']:6.0f} p={pick['price']:6.0f} part={pick['participation']:.4f} "
              f"risk={pick['exceed_risk']:.5f} cost={rows[-1]['cost_pp']:.2f}pp")
    ccg = pd.DataFrame(rows)
    ccg.to_csv(TAB / "final2_chance_grid.csv", index=False, encoding="utf-8-sig")
    S["chance_grid"] = ccg.to_dict("records")

    # ---------------- cap arithmetic for the abstract
    cap = 10000.
    row_cap = fr250[np.isclose(fr250.subsidy, cap)].iloc[0]
    d = 11250. - cap
    dp = best250["price"] - row_cap["price"]
    print(f"[D] cap row: s=10,000 p={row_cap['price']:.0f} part={row_cap['participation']:.4f}; "
          f"optimum s=11,250 p={best250['price']:.0f} part={best250['participation']:.4f}; "
          f"ds={d:.0f} dp={dp:.0f} share={dp/d:.3f}")
    S["cap"] = dict(s_cap=cap, p_cap=float(row_cap["price"]),
                    part_cap=float(row_cap["participation"]),
                    s_opt=float(best250["subsidy"]), p_opt=float(best250["price"]),
                    part_opt=float(best250["participation"]),
                    ds=float(d), dp=float(dp), share=float(dp / d),
                    dpart_pp=100 * (best250["participation"] - row_cap["participation"]))

    # ---------------- (D) instruments, method disclosed
    rows = []
    rows.append(dict(design="Joint choice (config + price)",
                     subsidy=float(best250["subsidy"]), price=float(best250["price"]),
                     configuration=best250["configuration"],
                     participation=float(best250["participation"]),
                     government=float(best250["government"]), method="250 CNY grid"))
    _, bf, _ = solve(D, base, dp=250, ds=250, allowed=["t1h_g8"])
    rows.append(dict(design="Resolution fixed at finest", subsidy=float(bf["subsidy"]),
                     price=float(bf["price"]), configuration=bf["configuration"],
                     participation=float(bf["participation"]),
                     government=float(bf["government"]), method="250 CNY grid"))
    _, bc, _ = solve(D, base, dp=250, ds=250, allowed=["t24h_g1"])
    rows.append(dict(design="Resolution fixed at coarsest", subsidy=float(bc["subsidy"]),
                     price=float(bc["price"]), configuration=bc["configuration"],
                     participation=float(bc["participation"]),
                     government=float(bc["government"]), method="250 CNY grid"))
    pf = 25500.
    _, bp1, _ = solve(D, base, dp=250, ds=1, prices=[pf])
    _, bp250, _ = solve(D, base, dp=250, ds=250, prices=[pf])
    rows.append(dict(design="Price fixed at 25,500 (budget boundary)",
                     subsidy=float(bp1["subsidy"]), price=float(bp1["price"]),
                     configuration=bp1["configuration"],
                     participation=float(bp1["participation"]),
                     government=float(bp1["government"]),
                     method="1 CNY bisection on the budget boundary"))
    rows.append(dict(design="Price fixed at 25,500 (250 CNY grid)", subsidy=float(bp250["subsidy"]),
                     price=float(bp250["price"]), configuration=bp250["configuration"],
                     participation=float(bp250["participation"]),
                     government=float(bp250["government"]), method="250 CNY grid"))
    inst = pd.DataFrame(rows)
    inst.to_csv(TAB / "final2_instruments.csv", index=False, encoding="utf-8-sig")
    print(inst.to_string(index=False))
    S["instruments"] = inst.to_dict("records")
    # budget-boundary check for the bisection value
    if bp1["subsidy"] > 0:
        Pb = expit((D[bp1["configuration"]]["b"] - bp1["price"] + bp1["subsidy"]) / base.sigma)
        S["boundary_check"] = dict(sub=float(bp1["subsidy"]),
                                   spending=float(bp1["subsidy"] * Pb.sum()),
                                   adopters=float(Pb.sum()))
        print("[D] boundary:", S["boundary_check"])

    S["seconds"] = time.perf_counter() - t0
    (OUT / "final2_summary.json").write_text(
        json.dumps(S, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v)),
        encoding="utf8")
    print("done in", round(S["seconds"], 1), "s")


if __name__ == "__main__":
    main()
