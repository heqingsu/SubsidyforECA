"""
create by hqs.
"""
from pathlib import Path
from dataclasses import dataclass, replace
import json, time
import numpy as np
import pandas as pd
from scipy.special import expit, logit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"; FIG = OUT / "figures"; TAB = OUT / "tables"
for d in (OUT, FIG, TAB):
    d.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})

E0 = pd.read_csv(ROOT / "data/enterprises.csv")
F0 = pd.read_csv(ROOT / "data/hourly_factors.csv")
phi0 = F0.pivot(index="hour", columns="region", values="factor").to_numpy()
w0 = F0.pivot(index="hour", columns="region", values="reference_weight").to_numpy()
configs = [(b, G, f"t{b}h_g{G}") for b in (24, 6, 1) for G in (1, 4, 8)]


@dataclass
class C:
    mu: float = .6
    lam: float = .5
    omega: float = 0.
    budget: float = 1e6
    pref: float = 1e6
    href: float = 1e7
    xref: float = 1e6
    sigma: float = 10000.
    eta: float = .8
    information: float = 0.
    cost_scale: float = 1.
    variable: float = 0.
    pmax: float = 60000.
    smax: float = 20000.
    tau: float = 1e-9


def build(df, phi, w, c, allowed=None):
    h = np.arange(24)
    profiles = np.array([.15 + ((h >= 8) & (h < 17)),
                         .15 + ((h >= 6) & (h < 22)),
                         np.ones(24)])
    load = profiles[df["shift"].to_numpy(int)]
    load = load / load.sum(1, keepdims=True)
    E = df.electricity_MWh.to_numpy()
    loc = df.region.to_numpy(int)
    q = E[:, None] * load
    base = E * np.sum(phi * w) / w.sum()
    out = {}
    for b, G, name in configs:
        if allowed is not None and name not in allowed:
            continue
        coarse = np.empty_like(phi)
        group = 8 // G
        for t in range(0, 24, b):
            for g in range(0, 8, group):
                sl = np.s_[t:t + b, g:g + group]
                coarse[sl] = np.sum(phi[sl] * w[sl]) / w[sl].sum()
        Crep = (q * coarse[:, loc].T).sum(1)
        D = base - Crep
        B = c.information + c.eta * df.eligible_fraction.to_numpy() * df.unit_value.to_numpy() * D
        net = B - df.extra_cost.to_numpy()
        fixed = c.cost_scale * (50000 + 12000 * (24 / b) ** .65 + 15000 * G ** .8)
        out[name] = dict(b=net, B=B, D=D, net_true=net, F=fixed,
                         fine=(q * phi[:, loc].T).sum(1), q=q, coarse=coarse, loc=loc)
    return out


def pbinom_exceed(p, thr):
    dist = np.zeros(len(p) + 1); dist[0] = 1.
    for pi in p:
        dist[1:] = dist[1:] * (1 - pi) + dist[:-1] * pi
        dist[0] *= (1 - pi)
    lo = min(max(int(thr) + 1, 0), len(p))
    return float(dist[lo:].sum())


def shock_exceed(p, thr, su, nodes=15):
    """Exceedance under a common logit-scale shock of size su (0 = independent)."""
    if su <= 0:
        return pbinom_exceed(p, thr)
    x, wq = np.polynomial.hermite_e.hermegauss(nodes)
    wq = wq / wq.sum()
    L = logit(np.clip(p, 1e-9, 1 - 1e-9))
    tot = 0.
    for xi, wi in zip(x, wq):
        tot += wi * pbinom_exceed(expit(L + su * xi), thr)
    return float(tot)


def solve(data, c, dp=250., ds=250., prices=None, allowed=None, risk_fn=None):
    st = time.perf_counter()
    pgrid = np.arange(0, c.pmax + dp / 2, dp) if prices is None else np.asarray(prices, float)
    sgrid = np.arange(0, c.smax + ds / 2, ds)
    sgrid = np.unique(np.r_[sgrid[sgrid <= c.smax], c.smax])
    names = list(data) if allowed is None else [n for n in allowed if n in data]
    cache = {}
    for name in names:
        d = data[name]
        zgrid = np.unique((pgrid[None, :] - sgrid[:, None]).ravel())
        prob = expit((d["b"][None, :] - zgrid[:, None]) / c.sigma)
        cache[name] = (zgrid, prob.sum(1), prob @ d["B"])
    recs = []
    for s in sgrid:
        cand = [dict(configuration="no_service", subsidy=float(s), price=0., participation=0.,
                     profit=0., spending=0., benefit=0., withdrawal=0., provider=0., government=0.)]
        incumbent = 0.
        for name in names:
            d = data[name]; n = len(d["b"])
            up_p = expit((d["b"] + s) / c.sigma)
            up_profit = max(c.pmax - c.variable, 0) * up_p.sum() - d["F"]
            up_obj = c.mu * up_p.mean() + (1 - c.mu) * up_profit / c.pref
            if up_profit < 0 or up_obj < incumbent - c.tau:
                continue
            zg, Aall, Hall = cache[name]
            idx = np.searchsorted(zg, pgrid - s)
            A, H = Aall[idx], Hall[idx]
            profit = (pgrid - c.variable) * A - d["F"]
            obj = c.mu * A / n + (1 - c.mu) * profit / c.pref
            ok = profit >= 0
            if not ok.any():
                continue
            vmax = obj[ok].max(); incumbent = max(incumbent, float(vmax))
            for j in np.where(ok & (obj >= vmax - c.tau))[0]:
                probs = expit((d["b"] - pgrid[j] + s) / c.sigma)
                off = expit((d["b"] - pgrid[j]) / c.sigma).mean()
                spend = s * A[j]
                gov = c.lam * H[j] / c.href + (1 - c.lam) * off - c.omega * spend / c.xref
                cand.append(dict(configuration=name, subsidy=float(s), price=float(pgrid[j]),
                                 participation=float(A[j] / n), profit=float(profit[j]),
                                 spending=float(spend), benefit=float(H[j]),
                                 withdrawal=float(off), provider=float(obj[j]), government=float(gov)))
        bestp = max(x["provider"] for x in cand)
        br = [x for x in cand if x["provider"] >= bestp - c.tau]
        aff = [x for x in br if x["spending"] <= c.budget + 1e-7]
        out = max(aff, key=lambda x: (x["government"], -x["price"])) if aff else \
            min(br, key=lambda x: x["spending"])
        out = dict(out); out["budget_feasible"] = bool(aff)
        recs.append(out)
    fr = pd.DataFrame(recs)
    fr["exceed_risk"] = np.nan
    feas = fr[fr.budget_feasible]
    best = feas.sort_values(["government", "subsidy"], ascending=[False, True]).iloc[0].to_dict()
    # net indicator (enterprise surplus net of subsidy and implementation cost);
    # the implementation cost is charged whenever a service is offered, including
    # zero-subsidy offers where the earlier guard silently dropped it
    if best["configuration"] != "no_service" and best["configuration"] in data:
        d = data[best["configuration"]]
        P = expit((d["b"] - best["price"] + best["subsidy"]) / c.sigma)
        rf = risk_fn or pbinom_exceed
        best["exceed_risk"] = (rf(P, np.floor(c.budget / best["subsidy"]))
                               if best["subsidy"] > 0 else 0.)
        best["impl_cost"] = float(np.sum(P * E0.extra_cost.to_numpy()))
        best["net_indicator"] = float(best["benefit"] - best["spending"] - best["impl_cost"])
    else:
        best["exceed_risk"] = 0.
        best["impl_cost"] = 0.
        best["net_indicator"] = 0.
    return fr, best, dict(seconds=time.perf_counter() - st)


def social(best, data, c, risk_fn=None):
    d = data[best["configuration"]]
    P = expit((d["b"] - best["price"] + best["subsidy"]) / c.sigma)
    thr = np.floor(c.budget / best["subsidy"]) if best["subsidy"] > 0 else 0
    rf = risk_fn or pbinom_exceed
    best = dict(best)
    best["exceed_risk"] = rf(P, thr) if best["subsidy"] > 0 else 0.
    best["impl_cost"] = float(np.sum(P * E0.extra_cost.to_numpy()))
    best["net_indicator"] = float(best["benefit"] - best["spending"] - best["impl_cost"])
    return best


def main():
    summary = {}
    base = C()
    D = build(E0, phi0, w0, base)

    # ---------- (1) grid convergence ----------
    rows = []
    for g in (500, 250, 100, 50):
        fr, best, info = solve(D, base, dp=g, ds=g)
        rows.append(dict(grid=g, subsidy=best["subsidy"], price=best["price"],
                         configuration=best["configuration"],
                         government=best["government"], participation=best["participation"],
                         spending=best["spending"], exceed=best["exceed_risk"],
                         risk_free=0.,
                         seconds=info["seconds"]))
        print("grid", g, round(best["subsidy"]), best["configuration"], round(best["participation"], 4),
              round(best["government"], 6), "risk", round(best["exceed_risk"], 5))
    conv = pd.DataFrame(rows)
    conv.to_csv(TAB / "final_grid_convergence.csv", index=False, encoding="utf-8-sig")
    summary["grid"] = conv.to_dict("records")

    # ---------- (2) candidate table around the optimum ----------
    fr250, best250, _ = solve(D, base, dp=250, ds=250)
    band = fr250[(fr250.subsidy >= 9750) & (fr250.subsidy <= 11500)].copy()
    for k in band.index:
        r = band.loc[k]
        if r.subsidy <= 0 or r.configuration == "no_service":
            band.loc[k, "exceed_risk"] = 0.
            continue
        dd = D[r.configuration]
        pp = expit((dd["b"] - r.price + r.subsidy) / base.sigma)
        band.loc[k, "exceed_risk"] = pbinom_exceed(pp, np.floor(base.budget / r.subsidy))
    band["status"] = np.where(band.spending > base.budget, "exceeds_expected_budget",
                              np.where(band.exceed_risk > 0.10, "violates_alpha10", "eligible"))
    # feasibility against the chance constraint
    sel = band[(band.exceed_risk <= 0.10) & (band.spending <= base.budget)]
    pick = sel.sort_values("government", ascending=False).iloc[0]
    band["selected_alpha10"] = band.subsidy == pick.subsidy
    band.to_csv(TAB / "final_candidate_band.csv", index=False, encoding="utf-8-sig")
    print("alpha10 pick", pick.subsidy, pick.price, round(pick.participation, 4),
          round(pick.exceed_risk, 5))
    summary["band"] = band[["subsidy", "price", "government", "participation", "spending",
                            "exceed_risk", "status", "selected_alpha10"]].to_dict("records")

    # ---------- (3) recognition regimes ----------
    rows = []
    for label, cc in [("No monetary recognition", replace(base, eta=0., information=0.)),
                      ("Information value only", replace(base, eta=0., information=5000.)),
                      ("Institutional recognition (baseline)", base),
                      ("Partial recognition (eta=0.4)", replace(base, eta=.4))]:
        DD = build(E0, phi0, w0, cc)
        _, bb, _ = solve(DD, cc, dp=250, ds=250)
        rows.append(dict(regime=label, subsidy=bb["subsidy"], price=bb["price"],
                         configuration=bb["configuration"], participation=bb["participation"],
                         profit=bb["profit"], spending=bb["spending"],
                         exceed=bb["exceed_risk"], net_indicator=bb["net_indicator"]))
        print(label, bb["configuration"], round(bb["subsidy"]), round(bb["participation"], 4))
    reg = pd.DataFrame(rows)
    reg.to_csv(TAB / "final_recognition_regimes.csv", index=False, encoding="utf-8-sig")
    summary["regimes"] = reg.to_dict("records")

    # ---------- (4) government-objective variants ----------
    rows = []
    for label, cc in [("Expected budget, fixed-offer withdrawal", base),
                      ("No withdrawal term (lambda=1)", replace(base, lam=1.0)),
                      ("Withdrawal weight 0.25", replace(base, lam=.75))]:
        DD = build(E0, phi0, w0, cc)
        _, bb, _ = solve(DD, cc, dp=250, ds=250)
        rows.append(dict(objective=label, subsidy=bb["subsidy"], price=bb["price"],
                         configuration=bb["configuration"], participation=bb["participation"],
                         spending=bb["spending"], benefit=bb["benefit"],
                         impl_cost=bb["impl_cost"], exceed=bb["exceed_risk"],
                         net_indicator=bb["net_indicator"]))
        print(label, round(bb["subsidy"]), round(bb["participation"], 4))
    # re-optimised withdrawal at s = 0 for the baseline offer
    s0 = best250["subsidy"]; p0 = best250["price"]
    reopt = None
    for name, d in D.items():
        P = expit((d["b"]) / base.sigma)
        prof = p0 * P.sum() - d["F"]
        if prof >= 0:
            pass
    fr0, best0, _ = solve(D, base, dp=250, ds=250)
    # provider problem at s = 0
    zero = fr0[fr0.subsidy == 0].iloc[0]
    _dz = D[zero.configuration]
    _Pz = expit((_dz["b"] - zero.price) / base.sigma)
    _implz = float(np.sum(_Pz * E0.extra_cost.to_numpy()))
    rows.append(dict(objective="Re-optimised offer after withdrawal (s=0)",
                     subsidy=0., price=zero.price, configuration=zero.configuration,
                     participation=zero.participation, spending=0., benefit=zero.benefit,
                     impl_cost=_implz, net_indicator=float(zero.benefit - _implz),
                     exceed=0.))
    obj = pd.DataFrame(rows)
    obj.to_csv(TAB / "final_objective_variants.csv", index=False, encoding="utf-8-sig")
    summary["objectives"] = obj.to_dict("records")

    # ---------- (5) common-shock risk ----------
    dsel = D[best250["configuration"]]
    P = expit((dsel["b"] - best250["price"] + best250["subsidy"]) / base.sigma)
    thr = np.floor(base.budget / best250["subsidy"])
    rows = []
    for su in (0., .5, 1.0, 1.5, 2.0):
        rows.append(dict(shock=su, risk=shock_exceed(P, thr, su)))
        print("shock", su, round(rows[-1]["risk"], 4))
    sh = pd.DataFrame(rows)
    sh.to_csv(TAB / "final_common_shock.csv", index=False, encoding="utf-8-sig")
    summary["shock"] = sh.to_dict("records")

    # ---------- (6) instrument benchmarks ----------
    rows = []
    rows.append(dict(design="Joint choice (config + price)", subsidy=best250["subsidy"],
                     price=best250["price"], configuration=best250["configuration"],
                     participation=best250["participation"], government=best250["government"]))
    _, bf, _ = solve(D, base, dp=250, ds=250, allowed=["t1h_g8"])
    rows.append(dict(design="Resolution fixed at finest", subsidy=bf["subsidy"], price=bf["price"],
                     configuration=bf["configuration"], participation=bf["participation"],
                     government=bf["government"]))
    _, bc, _ = solve(D, base, dp=250, ds=250, allowed=["t24h_g1"])
    rows.append(dict(design="Resolution fixed at coarsest", subsidy=bc["subsidy"], price=bc["price"],
                     configuration=bc["configuration"], participation=bc["participation"],
                     government=bc["government"]))
    pf = 25500.
    _, bp, _ = solve(D, base, dp=250, ds=1, prices=[pf])
    rows.append(dict(design="Price fixed at 25,500", subsidy=bp["subsidy"], price=bp["price"],
                     configuration=bp["configuration"], participation=bp["participation"],
                     government=bp["government"]))
    bench = pd.DataFrame(rows)
    bench.to_csv(TAB / "final_instrument_benchmarks.csv", index=False, encoding="utf-8-sig")
    print(bench.to_string())
    summary["benchmarks"] = bench.to_dict("records")

    # ---------- (7) resolution map with compliance value ----------
    def flat_phi(phi, alpha=.6):
        m = phi.mean(0)
        return alpha * m[None, :] + (1 - alpha) * phi

    def flat_w(w, alpha=.6):
        return np.ones_like(w)

    maps = []
    for tag, phi, w, c in [("compliance_value", phi0, w0, replace(base, information=2000.)),
                           ("flat_profile", flat_phi(phi0), flat_w(w0), replace(base, information=2000.)),
                           ("cost_json", phi0, w0, replace(base, information=2000.))]:
        mult = np.array([0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 15, 20, 30, 50])
        etas = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        grid = pd.DataFrame(index=[f"x{m}" for m in mult], columns=[f"e{e}" for e in etas])
        for m in mult:
            cc = replace(c, cost_scale=m)
            DD = build(E0, phi, w, cc)
            for e in etas:
                c2 = replace(cc, eta=e)
                DD2 = {k: dict(v, b=c2.information + c2.eta * E0.eligible_fraction.to_numpy()
                               * E0.unit_value.to_numpy() * v["D"] - E0.extra_cost.to_numpy())
                       for k, v in DD.items()}
                _, b2, _ = solve(DD2, c2, dp=1000, ds=500)
                grid.loc[f"x{m}", f"e{e}"] = b2["configuration"]
        grid.to_csv(TAB / f"final_map_{tag}.csv", encoding="utf-8-sig")
        maps.append((tag, grid))
        print(tag, "done")

    # plot: 2 panels
    order = ["t1h_g8", "t1h_g4", "t1h_g1", "t6h_g8", "t6h_g4", "t6h_g1", "t24h_g8", "t24h_g4", "t24h_g1", "no_service"]
    short = {"t1h_g8": "1h/8z", "t1h_g4": "1h/4z", "t1h_g1": "1h/1z",
             "t6h_g8": "6h/8z", "t6h_g4": "6h/4z", "t6h_g1": "6h/1z",
             "t24h_g8": "24h/8z", "t24h_g4": "24h/4z", "t24h_g1": "24h/1z", "no_service": "none"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    codes = {n: i for i, n in enumerate(order)}
    for ax, (tag, grid), title in zip(axes, maps[:2],
                                      ["(a) Baseline factor profile, V$_0$ = 2,000 CNY",
                                       "(b) Flattened factor profile, V$_0$ = 2,000 CNY"]):
        g = grid.apply(lambda col: col.map(lambda x: codes.get(x, len(order))))
        im = ax.imshow(g.to_numpy(float), aspect="auto", cmap="viridis", origin="lower",
                       vmin=0, vmax=len(order) - 1)
        ax.set_xticks(range(grid.shape[1])); ax.set_xticklabels([c[1:] for c in grid.columns])
        ax.set_yticks(range(grid.shape[0])); ax.set_yticklabels([i[1:] for i in grid.index])
        ax.set_xlabel("Recognition rate $\\eta$"); ax.set_ylabel("Cost multiplier")
        ax.set_title(title, fontsize=8)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                v = int(g.iat[i, j]); lab = "none" if v == len(order) else short[order[v]]
                ax.text(j, i, lab, ha="center", va="center", fontsize=5.2, color="w")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"final_resolution_map.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # capture transition counts for the text
    for tag, grid in maps[:2]:
        vals = grid.to_numpy().ravel().tolist()
        summary[f"map_{tag}_counts"] = {k: vals.count(k) for k in order}
        summary[f"map_{tag}_daily"] = sum(vals.count(k) for k in ("t24h_g8", "t24h_g4", "t24h_g1"))

    # ---------- attributable-change diagnostic (relabelled) ----------
    h = np.arange(24)
    rows = []
    rho = .2
    for name, d in D.items():
        rec = []
        for i in range(len(E0)):
            q = d["q"][i]; loc = int(E0.region.iloc[i]); fac = d["coarse"][:, loc]
            m = q > 0
            if m.sum() < 4:
                rec.append(0.); continue
            qq = q[m]; ff = fac[m]
            flex = rho * qq.sum(); share = qq / qq.sum()
            order_ = np.argsort(-ff)
            donor = order_[:max(1, len(order_) // 4)]
            recv = order_[-max(1, len(order_) // 4):]
            moved = np.minimum(flex * share[donor] / max(share[donor].sum(), 1e-12), qq[donor])
            moved = moved / max(moved.sum(), 1e-12) * flex
            dq = np.zeros_like(qq); dq[donor] -= moved
            add = np.minimum(np.full(len(recv), flex / len(recv)), qq[recv])
            add = add / max(add.sum(), 1e-12) * flex
            dq[recv] += add
            rec.append(float(-np.dot(dq, ff)))
        rows.append(dict(configuration=name, attributed_tCO2e=float(np.mean(rec) * len(E0)),
                         fixed_cost=float(d["F"])))
    att = pd.DataFrame(rows)
    att["cost_per_attributed_tonne"] = att.fixed_cost / att.attributed_tCO2e.replace(0, np.nan)
    att.to_csv(TAB / "final_attributed_change.csv", index=False, encoding="utf-8-sig")
    print(att.to_string())
    summary["attributed"] = att.to_dict("records")

    (OUT / "final_summary.json").write_text(
        json.dumps(summary, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v)),
        encoding="utf8")
    print("done")


if __name__ == "__main__":
    main()
