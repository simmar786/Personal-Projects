# Monte Carlo Delta-Hedging PnL Engine (European Call) with Rich Outputs
#
# - Self-financing accounting (option + stock hedge + cash)
# - Exposes detailed tables (DataFrames) and saved plots
# - Robust to scalar/array inputs
#


import os
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt


# ---------- Setup output dir ----------
BASE_DIR = os.path.join(os.path.dirname(__file__), "DeltaHedgeOutputs_v2")
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

# ---------- Black–Scholes helpers ----------
def bs_call_price(S, K, r, q, sigma, tau):
    S = np.asarray(S)
    tau = np.maximum(tau, 0.0)
    sqrt_tau = np.sqrt(np.where(tau > 0, tau, 1.0))
    denom = sigma * sqrt_tau + 1e-18
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / denom
    d2 = d1 - sigma * sqrt_tau
    return np.where(
        tau > 0,
        S * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2),
        np.maximum(S - K, 0.0)
    )

def bs_call_delta_spot(S, K, r, q, sigma, tau):
    S = np.asarray(S)
    tau = np.maximum(tau, 0.0)
    sqrt_tau = np.sqrt(np.where(tau > 0, tau, 1.0))
    denom = sigma * sqrt_tau + 1e-18
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / denom
    # Spot delta; at expiry becomes 1{S>K}
    return np.where(tau > 0, np.exp(-q * tau) * norm.cdf(d1), 1.0 * (S > K))

def futures_delta_from_spot_delta(spot_delta, r, q, tau):
    # Hedge using futures on the same maturity T
    # h = Δ_spot / (∂F/∂S) = Δ_spot / exp((r - q) * tau) = exp(-r * tau) * N(d1)
    return spot_delta / np.exp((r - q) * np.maximum(tau, 0.0))


# ---------- GBM path simulation ----------
def simulate_spot_paths(S0, mu_true, sigma_true, T, steps, n_paths, seed=0):
    rng = np.random.default_rng(seed)
    dt = T / steps
    Z = rng.standard_normal((n_paths, steps))
    incr = (mu_true - 0.5 * sigma_true**2) * dt + sigma_true * np.sqrt(dt) * Z
    log_S = np.cumsum(incr, axis=1)
    S = np.empty((n_paths, steps + 1))
    S[:, 0] = S0
    S[:, 1:] = S0 * np.exp(log_S)
    return S, dt


# ---------- Core delta-hedging engine (self-financing) ----------
def delta_hedge_run(
    S0=100.0, K=101.0,
    r=0.05, q=0.0,                 # carry / forward regime
    sigma_true=0.20,               # market vol for path gen
    sigma_model=None,              # model vol for hedging (defaults to sigma_true)
    mu_true=None,                  # market drift; default r - q (risk-neutral)
    T=1.0, steps=252, n_paths=50000, seed=7,
    hedge_instrument="spot",       # "spot" or "futures"
    simple_interest=False, save_prefix="baseline"
):
    if sigma_model is None:
        sigma_model = sigma_true
    if mu_true is None:
        mu_true = r - q            # risk-neutral drift by default

    # Simulate market paths under (mu_true, sigma_true)
    S, dt = simulate_spot_paths(S0, mu_true, sigma_true, T, steps, n_paths, seed)
    tau = np.linspace(T, 0.0, steps + 1)
    erdt = (1.0 + r * dt) if simple_interest else math.exp(r * dt)

    # t=0 model values for *each path*
    S0s = np.full(n_paths, S0)
    C0 = bs_call_price(S0s, K, r, q, sigma_model, tau[0])
    d0_spot = bs_call_delta_spot(S0s, K, r, q, sigma_model, tau[0])

    # Choose hedge object and initialize position
    if hedge_instrument == "spot":
        hedge = d0_spot.copy()     # this is Δ wrt spot (we short Δ shares)
    elif hedge_instrument == "futures":
        h0 = futures_delta_from_spot_delta(d0_spot, r, q, tau[0])   # Black-76 delta
        hedge = h0.copy()          # number of futures
    else:
        raise ValueError("hedge_instrument must be 'spot' or 'futures'.")

    # Self-financing setup: long option, hedge position, cash
    # For spot: cash = -C0 + Δ0 * S0 (since we *short* Δ shares, we receive Δ·S cash)
    # For futures: entering futures needs no cash → cash = -C0
    if hedge_instrument == "spot":
        cash = -C0 + hedge * S[:, 0]
    else:  # futures
        cash = -C0

    # Keep "previous" model option value per path
    Cprev = C0.copy()

    # Collectors
    option_pnl = np.zeros((n_paths, steps))
    hedge_pnl  = np.zeros((n_paths, steps))
    fund_pnl   = np.zeros((n_paths, steps))
    carry_pnl  = np.zeros((n_paths, steps)) if hedge_instrument == "spot" and q != 0.0 else None

    cash_series = np.zeros((n_paths, steps + 1))
    hedge_series = np.zeros((n_paths, steps + 1))
    option_val_series = np.zeros((n_paths, steps + 1))

    cash_series[:, 0] = cash
    hedge_series[:, 0] = hedge
    option_val_series[:, 0] = C0

    for t in range(steps):
        St0 = S[:, t]
        St1 = S[:, t + 1]
        tau1 = tau[t + 1]

        # Model revaluation at t+1
        Ct1 = bs_call_price(St1, K, r, q, sigma_model, tau1)
        option_pnl[:, t] = Ct1 - Cprev

        # Funding on pre-accrual cash
        fund_pnl[:, t] = cash * (erdt - 1.0)
        cash = cash * erdt

        # Hedging PnL + re-hedge logic
        if hedge_instrument == "spot":
            # hedge PnL from short-Δ over [t, t+1]
            hedge_pnl[:, t] = -hedge * (St1 - St0)

            # dividend carry on the stock position over the step (pay when short)
            if carry_pnl is not None:
                carry_pnl[:, t] = - q * hedge * St1 * dt
                cash += carry_pnl[:, t]

            # next hedge (spot delta)
            dnext_spot = bs_call_delta_spot(St1, K, r, q, sigma_model, tau1)

            # re-hedge cashflow: increasing short (Δ↑) means *sell* shares → cash inflow
            cash += (dnext_spot - hedge) * St1

            hedge = dnext_spot

        else:  # futures hedging
            # Futures for delivery at T: F(t,T) = S_t * exp((r - q)*(T - t))
            F0 = St0 * np.exp((r - q) * tau[t])
            F1 = St1 * np.exp((r - q) * tau1)

            # variation margin realized into cash immediately (after accrual this step)
            hedge_pnl[:, t] = -hedge * (F1 - F0)
            cash += hedge_pnl[:, t]

            # next futures delta (equiv to spot-delta divided by ∂F/∂S)
            dnext_spot = bs_call_delta_spot(St1, K, r, q, sigma_model, tau1)
            hnext = futures_delta_from_spot_delta(dnext_spot, r, q, tau1)
            # No re-hedge cashflow for futures (no inventory cost)
            hedge = hnext

        # roll and store
        Cprev = Ct1
        cash_series[:, t + 1] = cash
        hedge_series[:, t + 1] = hedge
        option_val_series[:, t + 1] = Ct1

    payoff = np.maximum(S[:, -1] - K, 0.0)

    # Final portfolio value and attribution reconciliation
    if hedge_instrument == "spot":
        final_value = payoff - hedge * S[:, -1] + cash
    else:  # futures
        final_value = payoff + cash

    # Total attribution sum (include carry if present)
    if carry_pnl is not None:
        total_attr = option_pnl.sum(axis=1) + hedge_pnl.sum(axis=1) + fund_pnl.sum(axis=1) + carry_pnl.sum(axis=1)
    else:
        total_attr = option_pnl.sum(axis=1) + hedge_pnl.sum(axis=1) + fund_pnl.sum(axis=1)

    consistency_error = final_value - total_attr

    # Build summary table (keep your original schema and add 'carry_pnl' if present)
    out_cols = {
        "total_pnl": total_attr,
        "option_pnl": option_pnl.sum(axis=1),
        "hedge_pnl": hedge_pnl.sum(axis=1),
        "fund_pnl": fund_pnl.sum(axis=1),
        "final_value": final_value,
        "consistency_error": consistency_error
    }
    if carry_pnl is not None:
        out_cols["carry_pnl"] = carry_pnl.sum(axis=1)

    dist_df = pd.DataFrame(out_cols)

    summ = {
        "S0": S0, "K": K, "r": r, "q": q,
        "sigma_true": sigma_true, "sigma_model": sigma_model,
        "mu_true": mu_true, "T": T, "steps": steps, "dt": dt, "n_paths": n_paths,
        "hedge_instrument": hedge_instrument,
        "interest_model": "simple(1+r*dt)" if simple_interest else "continuous(exp(r*dt))",
        "mean_total": float(dist_df["total_pnl"].mean()),
        "std_total": float(dist_df["total_pnl"].std()),
        "p05_total": float(dist_df["total_pnl"].quantile(0.05)),
        "p50_total": float(dist_df["total_pnl"].quantile(0.50)),
        "p95_total": float(dist_df["total_pnl"].quantile(0.95)),
        "max_abs_consistency_error": float(np.max(np.abs(dist_df["consistency_error"]))),
    }
    summary_df = pd.DataFrame([summ])

    # Save artifacts (your existing code below can remain unchanged)
    dist_path = os.path.join(BASE_DIR, f"{save_prefix}_distribution.csv")
    dist_df.to_csv(dist_path, index=False)

    summary_path = os.path.join(BASE_DIR, f"{save_prefix}_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # Display to user
    print("Delta-Hedging PnL Distribution (sample head)")
    print(dist_df.head(10))
    print("Delta-Hedging Summary")
    print(summary_df.head(10))

    # Plots
    # 1) Histogram of total PnL
    plt.figure()
    plt.hist(dist_df["total_pnl"].values, bins=100, density=True)
    plt.title("Total PnL Distribution")
    plt.xlabel("PnL")
    plt.ylabel("Density")
    fig1_path = os.path.join(BASE_DIR, f"{save_prefix}_pnl_hist.png")
    plt.savefig(fig1_path, bbox_inches="tight")
    plt.close()

    # 2) QQ-like normal probability plot of PnL (simple manual via sorted vs z)
    # (keeping it single axis and basic)
    vals = np.sort(dist_df["total_pnl"].values)
    n = len(vals)
    # Avoid exactly 0/1 for percentiles
    probs = (np.arange(1, n+1) - 0.5) / n
    z = norm.ppf(probs)
    plt.figure()
    plt.plot(z, vals, marker="", linestyle="-")
    plt.title("Total PnL Normal-QQ (heuristic)")
    plt.xlabel("Normal Quantiles")
    plt.ylabel("Sorted PnL")
    fig2_path = os.path.join(BASE_DIR, f"{save_prefix}_pnl_qq.png")
    plt.savefig(fig2_path, bbox_inches="tight")
    plt.close()

    # 3) Example single path spot + option value (two separate figures to respect one chart per figure)
    idx_sample = 0
    tgrid = np.arange(steps + 1) * (T / steps)

    plt.figure()
    plt.plot(tgrid, S[idx_sample, :])
    plt.title("Sample Path: Spot")
    plt.xlabel("Time (years)")
    plt.ylabel("Spot")
    fig3_path = os.path.join(BASE_DIR, f"{save_prefix}_sample_spot.png")
    plt.savefig(fig3_path, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(tgrid, option_val_series[idx_sample, :])
    plt.title("Sample Path: Option Value")
    plt.xlabel("Time (years)")
    plt.ylabel("Option value")
    fig4_path = os.path.join(BASE_DIR, f"{save_prefix}_sample_opt_val.png")
    plt.savefig(fig4_path, bbox_inches="tight")
    plt.close()

    # 4) Convergence plot scaffold: std vs step-size (we'll compute below in sweep); here save placeholder
    # Return everything needed for external convergence sweep
    return {
        "dist_csv": dist_path,
        "summary_csv": summary_path,
        "hist_png": os.path.join(BASE_DIR, f"{save_prefix}_pnl_hist.png"),
        "qq_png": os.path.join(BASE_DIR, f"{save_prefix}_pnl_qq.png"),
        "sample_spot_png": os.path.join(BASE_DIR, f"{save_prefix}_sample_spot.png"),
        "sample_optval_png": os.path.join(BASE_DIR, f"{save_prefix}_sample_opt_val.png"),
        "series": {"S": S, "cash": cash_series, "hedge": hedge_series, "optval": option_val_series},
        "distribution_df_head": dist_df.head(10),
        "summary_df": summary_df
    }

# ---------- Convergence sweep (mean -> 0, std -> 0) ----------
def convergence_sweep(
    S0=100.0, K=101.0,
    r=0.05, q=0.0,
    sigma_true=0.20,            # market vol for path gen
    sigma_model=None,           # hedge vol; defaults to sigma_true if None
    mu_true=None,               # market drift; defaults to r - q if None
    T=1.0, M=252, n_paths=100000, seed=9,
    freqs=(1, 2, 4, 12, 52, 252),
    hedge_instrument="spot",    # "spot" or "futures"
    simple_interest=False, save_prefix="convergence"
):
    if sigma_model is None:
        sigma_model = sigma_true
    if mu_true is None:
        mu_true = r - q

    rows = []
    for n_per_day in freqs:
        steps = int(M * n_per_day)
        out = delta_hedge_run(
            S0=S0, K=K,
            r=r, q=q,
            sigma_true=sigma_true,
            sigma_model=sigma_model,
            mu_true=mu_true,
            T=T, steps=steps, n_paths=n_paths, seed=seed,
            hedge_instrument=hedge_instrument,
            simple_interest=simple_interest,
            save_prefix=f"{save_prefix}_{hedge_instrument}_{n_per_day}x"
        )
        df = out["summary_df"]
        rows.append({
            "hedge_instrument": hedge_instrument,
            "sigma_true": float(df["sigma_true"].iloc[0]),
            "sigma_model": float(df["sigma_model"].iloc[0]),
            "q": float(df["q"].iloc[0]),
            "mu_true": float(df["mu_true"].iloc[0]),
            "n_per_day": n_per_day,
            "steps": steps,
            "dt": T / steps,
            "mean_total": float(df["mean_total"].iloc[0]),
            "std_total": float(df["std_total"].iloc[0]),
            "max_abs_consistency_error": float(df["max_abs_consistency_error"].iloc[0]),
        })

    conv_df = pd.DataFrame(rows).sort_values("dt", ascending=False).reset_index(drop=True)

    # Save & display
    conv_path = os.path.join(BASE_DIR, f"{save_prefix}_{hedge_instrument}_table.csv")
    conv_df.to_csv(conv_path, index=False)
    print("Convergence Table (Mean -> 0, Std -> 0)")
    print(conv_df.head(10))

    # Plot std vs dt (log-log)
    plt.figure()
    plt.loglog(conv_df["dt"].values, conv_df["std_total"].values, marker="o", linestyle="-")
    plt.title(f"Convergence: Std(PnL) vs dt [{hedge_instrument}]")
    plt.xlabel("dt")
    plt.ylabel("Std of Total PnL")
    fig_std_path = os.path.join(BASE_DIR, f"{save_prefix}_{hedge_instrument}_std_vs_dt.png")
    plt.savefig(fig_std_path, bbox_inches="tight")
    plt.close()

    # (Optional) Plot mean vs dt to verify it → 0
    plt.figure()
    plt.semilogx(conv_df["dt"].values, conv_df["mean_total"].values, marker="o", linestyle="-")
    plt.title(f"Convergence: Mean(PnL) vs dt [{hedge_instrument}]")
    plt.xlabel("dt")
    plt.ylabel("Mean Total PnL")
    fig_mean_path = os.path.join(BASE_DIR, f"{save_prefix}_{hedge_instrument}_mean_vs_dt.png")
    plt.savefig(fig_mean_path, bbox_inches="tight")
    plt.close()

    return {
        "conv_csv": conv_path,
        "conv_std_png": fig_std_path,
        "conv_mean_png": fig_mean_path,
        "conv_df": conv_df
    }


# ---------- Run a baseline experiment and a convergence sweep ----------

N_PATHS = 50000

baseline = delta_hedge_run(
    S0=100.0, K=101.0,
    r=0.05, q=0.00,
    sigma_true=0.20, sigma_model=0.50,   # match for replication
    mu_true=0.05,                        # or r - q
    T=1.0, steps=252, n_paths=N_PATHS, seed=123,
    hedge_instrument="spot",
    simple_interest=False, save_prefix="baseline_exp"
)

conv = convergence_sweep(
    S0=100.0, K=101.0,
    r=0.05, q=0.00,
    sigma_true=0.20, sigma_model=0.50,   # keep equal to test convergence
    mu_true=0.05,                        # or r - q
    T=1.0, M=252, n_paths=N_PATHS, seed=99,
    freqs=(1, 2, 4, 12),
    hedge_instrument="spot",
    simple_interest=False, save_prefix="conv_exp"
)

# Return the file paths for downloading
baseline_paths = {
    "distribution":       baseline["dist_csv"],
    "summary":            baseline["summary_csv"],
    "histogram":          baseline["hist_png"],
    "qq_plot":            baseline["qq_png"],
    "sample_spot":        baseline["sample_spot_png"],
    "sample_option_value":baseline["sample_optval_png"],
}

# convergence_sweep now returns: conv_csv, conv_std_png, conv_mean_png, conv_df
conv_paths = {
    "convergence_table":  conv["conv_csv"],
    "std_vs_dt_plot":     conv["conv_std_png"],
    "mean_vs_dt_plot":    conv["conv_mean_png"],
}

print(baseline_paths)
#print(conv_paths)
print("done")