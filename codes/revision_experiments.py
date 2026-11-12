"""
create by hqs

All monetary units are CNY per year.  Solver scope: finite grids, no certified
continuous global optimum.
"""
from pathlib import Path
from dataclasses import dataclass, replace, asdict
import json, time, platform, hashlib
import numpy as np
import pandas as pd
from scipy.special import expit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"; FIG = OUT / "figures"; TAB = OUT / "tables"
for d in (OUT, FIG, TAB):
    d.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})


@dataclass
class Settings:
    mu: float = .6          # coverage weight in the provider objective
    lam: float = .5         # weight on monetised accounting benefit
    omega: float = 0.       # fiscal weight (0 reproduces the original degenerate run)
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
    smax: float = 20000.    # cap no longer binding under omega > 0
    tau: float = 1e-9


E0 = pd.read_csv(ROOT / "data/synthetic_enterprises.csv")
F = pd.read_csv(ROOT / "data/synthetic_hourly_factors.csv")
phi = F.pivot(index="hour", columns="region", values="factor").to_numpy()
weights = F.pivot(index="hour", columns="region", values="reference_weight").to_numpy()
configs = [(b, G, f"t{b}h_g{G}") for b in (24, 6, 1) for G in (1, 4, 8)]

checks = []


def check(name, condition, detail=""):
    checks.append({"check": name, "passed": bool(condition), "detail": str(detail)})


def write(df, name):
    df.to_csv(TAB / f"{name}.csv", index=False, encoding="utf-8-sig")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def prepare(df, c):
    h = np.arange(24)
    profiles = np.array([.15 + ((h >= 8) & (h < 17)),
                         .15 + ((h >= 6) & (h < 22)),
                         np.ones(24)])
    load = profiles[df["shift"].to_numpy(int)]
    load = load / load.sum(1, keepdims=True)
    E = df.electricity_MWh.to_numpy()
    loc = df.region.to_numpy(int)
    q = E[:, None] * load
    baseline = E * np.sum(phi * weights) / weights.sum()
    fine = np.sum(q * phi[:, loc].T, axis=1)
    result = {}
    rows = []
    for b, G, name in configs:
        coarse = np.empty_like(phi)
        group = 8 // G
        for t in range(0, 24, b):
            for g in range(0, 8, group):
                sl = np.s_[t:t + b, g:g + group]
                coarse[sl] = np.sum(phi[sl] * weights[sl]) / weights[sl].sum()
        C = (q * coarse[:, loc].T).sum(1)
        D = baseline - C
        B = c.information + c.eta * df.eligible_fraction.to_numpy() * df.unit_value.to_numpy() * D
        net = B - df.extra_cost.to_numpy()
        fixed = c.cost_scale * (50000 + 12000 * (24 / b) ** .65 + 15000 * G ** .8)
        result[name] = {"b": net, "B": B, "D": D, "C": C, "F": fixed,
                        "fine": fine, "q": q, "coarse": coarse, "loc": loc}
        err = np.abs(C - fine) / np.maximum(fine, 1e-12)
        rows.append(dict(configuration=name, hours=b, zones=G, fixed_cost=fixed,
                         mean_signed_tonnes=D.mean(),
                         lower_footprint_share=np.mean(D > 1e-8),
                         weighted_deviation=np.dot(E, err) / E.sum()))
    return result, pd.DataFrame(rows)


def poisson_binomial_exceed(p, threshold):
    """P(sum of independent Bernoulli(p) > threshold), exact by convolution."""
    dist = np.zeros(len(p) + 1)
    dist[0] = 1.0
    for pi in p:
        dist[1:] = dist[1:] * (1 - pi) + dist[:-1] * pi
        dist[0] *= (1 - pi)
    lo = min(max(int(threshold) + 1, 0), len(p))
    return float(dist[lo:].sum())


def solve(data, c, dp=250., ds=250., allowed=None):
    """Enumerated grid optimum with an explicit fiscal weight and budget-risk output."""
    start = time.perf_counter()
    pgrid = np.arange(0, c.pmax + dp / 2, dp)
    pgrid = np.unique(np.r_[pgrid[pgrid <= c.pmax], c.pmax])
    sgrid = np.arange(0, c.smax + ds / 2, ds)
    sgrid = np.unique(np.r_[sgrid[sgrid <= c.smax], c.smax])
    names = list(data) if allowed is None else allowed
    cache = {}
    for name in names:
        d = data[name]
        zgrid = np.unique((pgrid[None, :] - sgrid[:, None]).ravel())
        prob = expit((d["b"][None, :] - zgrid[:, None]) / c.sigma)
        cache[name] = (zgrid, prob.sum(1), prob @ d["B"])
    records = []
    for s in sgrid:
        candidate = [dict(configuration="no_service", subsidy=float(s), price=0.,
                          effective_price=0., participation=0., profit=0.,
                          spending=0., benefit=0., withdrawal=0.,
                          withdrawal_profit=0., provider=0., government=0.)]
        incumbent = 0.
        for name in names:
            d = data[name]
            n = len(d["b"])
            upperprob = expit((d["b"] + s) / c.sigma)
            up_profit = max(c.pmax - c.variable, 0) * upperprob.sum() - d["F"]
            up_obj = c.mu * upperprob.mean() + (1 - c.mu) * up_profit / c.pref
            if up_profit < 0 or up_obj < incumbent - c.tau:
                continue
            zg, Aall, Hall = cache[name]
            idx = np.searchsorted(zg, pgrid - s)
            A = Aall[idx]; H = Hall[idx]
            profit = (pgrid - c.variable) * A - d["F"]
            obj = c.mu * A / n + (1 - c.mu) * profit / c.pref
            viable = profit >= 0
            if not viable.any():
                continue
            vmax = obj[viable].max()
            incumbent = max(incumbent, float(vmax))
            for j in np.where(viable & (obj >= vmax - c.tau))[0]:
                probs = expit((d["b"] - pgrid[j] + s) / c.sigma)
                off = expit((d["b"] - pgrid[j]) / c.sigma).mean()
                spending = s * A[j]
                gov = (c.lam * H[j] / c.href + (1 - c.lam) * off
                       - c.omega * spending / c.xref)
                risk = (poisson_binomial_exceed(probs, np.floor(c.budget / s))
                        if s > 0 else 0.0)
                candidate.append(dict(configuration=name, subsidy=float(s),
                                      price=float(pgrid[j]),
                                      effective_price=float(pgrid[j] - s),
                                      participation=float(A[j] / n),
                                      profit=float(profit[j]),
                                      spending=float(spending), benefit=float(H[j]),
                                      withdrawal=float(off),
                                      withdrawal_profit=float((pgrid[j] - c.variable) * off * n - d["F"]),
                                      provider=float(obj[j]), government=float(gov),
                                      exceed_risk=risk))
        bestprovider = max(x["provider"] for x in candidate)
        br = [x for x in candidate if x["provider"] >= bestprovider - c.tau]
        affordable = [x for x in br if x["spending"] <= c.budget + 1e-7]
        if affordable:
            out = max(affordable, key=lambda x: (x["government"], -x["price"]))
        else:
            out = min(br, key=lambda x: x["exceed_risk"])
        out = out.copy()
        out["budget_feasible"] = bool(affordable)
        out["response_count"] = len(br)
        records.append(out)
    frame = pd.DataFrame(records)
    feasible = frame[frame.budget_feasible]
    best = feasible.sort_values(["government", "subsidy"], ascending=[False, True]).iloc[0].to_dict()
    return frame, best, dict(seconds=time.perf_counter() - start, dp=dp, ds=ds)


def recognised_reduction(data, df, rho=0.2):
    """Emission reduction that a fixed physical load shift makes *visible* at each resolution.

    Enterprises move a share rho of annual electricity out of the highest-factor
    hours of their shift window into the lowest-factor hours, one-for-one in energy.
    A configuration can only credit the part of that shift that changes the
    reported footprint, i.e. variation surviving its own temporal-spatial averaging.
    """
    rows = []
    fine = data["t1h_g8"]
    h = np.arange(24)
    for name, d in data.items():
        recognised = []
        for i in range(len(df)):
            q = d["q"][i]; loc = int(df["region"].iloc[i])
            fac = d["coarse"][:, loc]
            m = q > 0
            if m.sum() < 4:
                recognised.append(0.); continue
            qq = q[m]; ff = fac[m]
            flex = rho * qq.sum()
            share = qq / qq.sum()
            order = np.argsort(-ff)
            donor = order[:max(1, len(order) // 4)]
            recv = order[-max(1, len(order) // 4):]
            moved = np.minimum(flex * share[donor] / max(share[donor].sum(), 1e-12), qq[donor])
            moved = moved / max(moved.sum(), 1e-12) * flex
            dq = np.zeros_like(qq)
            dq[donor] -= moved
            cap = np.maximum(qq[recv] - 0, 0)
            add = np.minimum(np.full(len(recv), flex / len(recv)), cap)
            add = add / max(add.sum(), 1e-12) * flex
            dq[recv] += add
            recognised.append(float(-np.dot(dq, ff)))
        rows.append(dict(configuration=name,
                         recognised_tCO2=float(np.mean(recognised) * len(df)),
                         mean_per_enterprise_tCO2=float(np.mean(recognised)),
                         fixed_cost=float(d["F"])))
    out = pd.DataFrame(rows)
    out["cost_per_tonne"] = np.where(out.recognised_tCO2 > 1e-9,
                                     out.fixed_cost / out.recognised_tCO2, np.nan)
    return out


def main():
    c0 = Settings()  # omega = 0 reproduces the original corner solution
    data, agg = prepare(E0, c0)
    write(agg, "aggregation")

    # ---- A. fiscal weight sweep: corner -> interior -> no subsidy -------------
    omegas = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.2, 0.5]
    rows = []
    for w in omegas:
        cw = replace(c0, omega=w)
        _, be, _ = solve(data, cw, 250, 250)
        rows.append(dict(omega=w, **{k: be[k] for k in
                                     ("configuration", "subsidy", "price",
                                      "participation", "profit", "spending",
                                      "government", "withdrawal", "exceed_risk")}))
    sweep = pd.DataFrame(rows)
    write(sweep, "omega_sweep")
    print(sweep.to_string(), flush=True)

    # Baseline regime for the paper: a fiscal weight that makes the optimum interior.
    interior = sweep[(sweep.subsidy > 0) & (sweep.subsidy < c0.smax - 1e-9)]
    check("interior optimum exists", len(interior) > 0, interior.omega.tolist())
    w_star = float(interior.omega.iloc[0]) if len(interior) else 0.02
    c = replace(c0, omega=w_star)
    print("selected baseline omega =", w_star, flush=True)

    base, best, perf = solve(data, c, 250, 250)
    write(base, "subsidy_response")
    print("BASE", best, flush=True)
    check("baseline not at cap", best["subsidy"] < c.smax - 1e-9, best["subsidy"])
    check("baseline budget feasible", best["spending"] <= c.budget + 1e-6)

    # ---- B. chance-constrained selection rule --------------------------------
    alphas = [0.50, 0.30, 0.20, 0.10, 0.05]
    rows = []
    for a in alphas:
        ok = base[(base.budget_feasible) & (base.exceed_risk <= a)]
        pick = ok.sort_values(["government", "subsidy"], ascending=[False, True]).iloc[0] if len(ok) else None
        if pick is None:
            rows.append(dict(alpha=a, configuration="none", subsidy=np.nan,
                             price=np.nan, participation=0., spending=0.,
                             government=np.nan, exceed_risk=np.nan, profit=np.nan))
        else:
            rows.append(dict(alpha=a, **{k: pick[k] for k in
                                         ("configuration", "subsidy", "price",
                                          "participation", "spending", "government",
                                          "exceed_risk", "profit")}))
    cc = pd.DataFrame(rows)
    write(cc, "chance_constrained")
    print(cc.to_string(), flush=True)

    # ---- C. resolution-selection map over cost and recognition ---------------
    cost_scales = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
    etas = [0.2, 0.4, 0.6, 0.8, 1.0]
    rows = []
    for cs in cost_scales:
        for et in etas:
            ck = replace(c, cost_scale=cs, eta=et)
            dd, _ = prepare(E0, ck)
            _, be, _ = solve(dd, ck, 500, 500)
            rows.append(dict(cost_scale=cs, eta=et, configuration=be["configuration"],
                             subsidy=be["subsidy"], participation=be["participation"],
                             government=be["government"]))
        print("map cost_scale", cs, flush=True)
    rmap = pd.DataFrame(rows)
    write(rmap, "resolution_map")
    check("resolution map varies", rmap.configuration.nunique() > 1, rmap.configuration.nunique())

    # ---- D. recognised emission reduction by configuration -------------------
    rec = recognised_reduction(data, E0)
    write(rec, "recognised_reduction")
    print(rec.to_string(), flush=True)

    # ---- E. robust (1000-resample) paired bootstrap --------------------------
    boots = []
    for seed in range(20261016, 20262016):
        rng = np.random.default_rng(seed)
        sample = E0.iloc[rng.integers(0, len(E0), len(E0))].reset_index(drop=True)
        dd, _ = prepare(sample, c)
        _, be, _ = solve(dd, c, 500, 500)
        boots.append(dict(seed=seed, **{k: be[k] for k in
                                        ("configuration", "subsidy", "price",
                                         "participation", "profit", "spending",
                                         "government", "exceed_risk")}))
    boot = pd.DataFrame(boots)
    write(boot, "bootstrap_1000")
    ci = {col: [float(boot[col].quantile(.025)), float(boot[col].quantile(.975))]
          for col in ("subsidy", "price", "participation", "government")}
    ci["configuration_share"] = boot.configuration.value_counts(normalize=True).to_dict()
    print("BOOTSTRAP CI", json.dumps(ci, ensure_ascii=False), flush=True)
    check("bootstrap resamples", len(boot) == 1000, len(boot))

    # ---- F. figures ----------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ax[0].plot(sweep.omega, sweep.subsidy / 1000, "o-")
    ax[0].axhline(c0.smax / 1000, ls=":", color="grey")
    ax[0].set(xlabel=r"Fiscal weight $\omega$", ylabel="Selected subsidy (thousand CNY)",
              title="(a) Subsidy schedule")
    ax[1].plot(sweep.omega, sweep.participation * 100, "o-", label="Participation")
    ax[1].plot(sweep.omega, sweep.withdrawal * 100, "s--", label="After withdrawal")
    ax[1].set(xlabel=r"Fiscal weight $\omega$", ylabel="Percent", title="(b) Adoption")
    ax[1].legend(fontsize=7)
    ax[2].plot(sweep.omega, sweep.spending / 1e6, "o-", label="Expected expenditure")
    ax[2].axhline(c.budget / 1e6, ls=":", color="grey", label="Budget")
    ax[2].set(xlabel=r"Fiscal weight $\omega$", ylabel="Million CNY", title="(c) Expenditure")
    ax[2].legend(fontsize=7)
    save(fig, "rev01_omega_sweep")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ok = cc.dropna(subset=["subsidy"])
    ax[0].plot(ok.alpha * 100, ok.participation * 100, "o-")
    ax[0].invert_xaxis(); ax[0].set(xlabel="Risk tolerance (%)", ylabel="Participation (%)",
                                    title="(a) Participation cost of risk control")
    ax[1].plot(ok.alpha * 100, ok.spending / 1e6, "o-", label="Expected expenditure")
    ax[1].plot(ok.alpha * 100, ok.subsidy / 1000, "s--", label="Subsidy (thousand CNY)")
    ax[1].invert_xaxis(); ax[1].legend(fontsize=7)
    ax[1].set(xlabel="Risk tolerance (%)", title="(b) Selected instruments")
    save(fig, "rev02_chance_constrained")

    codes = {n: i for i, n in enumerate([x[2] for x in configs])}
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    grid = rmap.pivot(index="eta", columns="cost_scale", values="configuration").replace(codes)
    im = ax.imshow(grid, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(cost_scales)), [f"{x:g}" for x in cost_scales])
    ax.set_yticks(range(len(etas)), [f"{x:g}" for x in etas])
    ax.set(xlabel="Cost multiplier", ylabel=r"Monetary recognition rate $\eta$",
           title="Selected configuration")
    for i in range(len(etas)):
        for j in range(len(cost_scales)):
            ax.text(j, i, list(codes)[int(grid.iloc[i, j])], ha="center", va="center",
                    fontsize=6, color="w")
    save(fig, "rev03_resolution_map")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    o = rec.sort_values("recognised_tCO2")
    ax[0].barh(o.configuration, o.recognised_tCO2, color="#2f6f5e")
    ax[0].set(xlabel="Recognised reduction (tCO2/year)", title="(a) Attribution capability")
    ax[1].barh(o.configuration, o.cost_per_tonne, color="#a4552f")
    ax[1].set(xlabel="Fixed cost per recognised tonne (CNY/tCO2)", title="(b) Unit cost")
    save(fig, "rev04_recognised_reduction")

    # ---- G. summary ----------------------------------------------------------
    summary = {"settings": asdict(c), "baseline": best,
               "omega_of_baseline": w_star,
               "bootstrap_ci": ci,
               "all_checks_passed": all(x["passed"] for x in checks),
               "versions": {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__},
               "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (ROOT / "data").glob("*.csv")}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                                 default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    write(pd.DataFrame(checks), "validation")
    assert all(x["passed"] for x in checks), [x for x in checks if not x["passed"]]
    print("ALL CHECKS PASSED", len(checks), flush=True)


if __name__ == "__main__":
    main()
