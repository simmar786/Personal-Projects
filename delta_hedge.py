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
BASE_DIR = os.path.join(os.path.dirname(__file__), "DeltaHedgeCallPnLDist")
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

# ---------- Black–Scholes helpers ----------
def bs_call_price(S, K, r, sigma, tau):
    S = np.asarray(S)
    tau = np.maximum(tau, 0.0)
    sqrt_tau = np.sqrt(np.where(tau > 0, tau, 1.0))
    denom = sigma * sqrt_tau + 1e-18
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / denom
    d2 = d1 - sigma * sqrt_tau
    # price collapses to payoff at expiry
    price = np.where(tau > 0, S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2),
                     np.maximum(S - K, 0.0))
    return price

def bs_call_delta(S, K, r, sigma, tau):
    S = np.asarray(S)
    tau = np.maximum(tau, 0.0)
    sqrt_tau = np.sqrt(np.where(tau > 0, tau, 1.0))
    denom = sigma * sqrt_tau + 1e-18
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / denom
    # At expiry, delta is 1 if S>K else 0 (ITM indicator)
    return np.where(tau > 0, norm.cdf(d1), 1.0 * (S > K))

# ---------- GBM path simulation ----------
def simulate_spot_paths(S0, r, sigma, T, steps, n_paths, seed=0):
    rng = np.random.default_rng(seed)
    dt = T / steps
    Z = rng.standard_normal((n_paths, steps))
    incr = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_S = np.cumsum(incr, axis=1)
    S = np.empty((n_paths, steps + 1))
    S[:, 0] = S0
    S[:, 1:] = S0 * np.exp(log_S)
    return S, dt

# ---------- Core delta-hedging engine (self-financing) ----------
def delta_hedge_run(S0=100.0, K=101.0, r=0.05, sigma=0.20, T=1.0, steps=252, n_paths=20000, seed=7,
                    simple_interest=False, save_prefix="baseline"):
    S, dt = simulate_spot_paths(S0, r, sigma, T, steps, n_paths, seed)
    tau = np.linspace(T, 0.0, steps + 1)  # remaining time grid
    erdt = (1.0 + r*dt) if simple_interest else math.exp(r * dt)

    # t=0 initialization
    S0s = np.full(n_paths, S0)
    C0 = bs_call_price(S0s, K, r, sigma, tau[0])
    d0 = bs_call_delta(S0s, K, r, sigma, tau[0])

    cash  = -C0 + d0 * S[:, 0]
    delta = d0.copy()
    Cprev = C0.copy()
    #delta = np.full(n_paths, d0)
    #Cprev = np.full(n_paths, C0)

    # Collectors
    option_pnl = np.zeros((n_paths, steps))
    hedge_pnl  = np.zeros((n_paths, steps))
    fund_pnl   = np.zeros((n_paths, steps))
    cash_series = np.zeros((n_paths, steps+1))
    delta_series = np.zeros((n_paths, steps+1))
    option_val_series = np.zeros((n_paths, steps+1))

    cash_series[:, 0] = cash
    delta_series[:, 0] = delta
    option_val_series[:, 0] = C0

    for t in range(steps):
        St0 = S[:, t]
        St1 = S[:, t + 1]
        Ct1 = bs_call_price(St1, K, r, sigma, tau[t + 1])

        # Attributions per step
        option_pnl[:, t] = Ct1 - Cprev
        hedge_pnl[:,  t] = -delta * (St1 - St0)
        fund_pnl[:,   t] = cash * (erdt - 1.0)

        # Cash accrual then rebalance
        cash = cash * erdt
        dnext = bs_call_delta(St1, K, r, sigma, tau[t + 1])
        cash += (dnext - delta) * St1

        # roll state
        delta = dnext
        Cprev = Ct1

        # store series
        cash_series[:, t+1] = cash
        delta_series[:, t+1] = delta
        option_val_series[:, t+1] = Ct1

    idx_sample = 0
    portfolio_series = (option_val_series[idx_sample]
                        - delta_series[idx_sample] * S[idx_sample]
                        + cash_series[idx_sample])
    plt.figure()
    plt.plot(np.arange(steps+1) * (T/steps), portfolio_series)
    plt.title("Portfolio value through time (should track cumulative PnL)")
    plt.xlabel("Time (years)")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f"{save_prefix}_sample_portfolio_value.png"))
    plt.close()

    payoff = np.maximum(S[:, -1] - K, 0.0)
    final_value = payoff - delta * S[:, -1] + cash

    total_attr = option_pnl.sum(axis=1) + hedge_pnl.sum(axis=1) + fund_pnl.sum(axis=1)
    consistency_error = final_value - total_attr

    # Build summary DataFrame for distribution
    dist_df = pd.DataFrame({
        "total_pnl": total_attr,
        "option_pnl": option_pnl.sum(axis=1),
        "hedge_pnl": hedge_pnl.sum(axis=1),
        "fund_pnl": fund_pnl.sum(axis=1),
        "final_value": final_value,
        "consistency_error": consistency_error
    })

    # Summary stats
    summ = {
        "S0": S0, "K": K, "r": r, "sigma": sigma, "T": T,
        "steps": steps, "dt": dt, "n_paths": n_paths,
        "interest_model": "simple(1+r*dt)" if simple_interest else "continuous(exp(r*dt))",
        "mean_total": float(dist_df["total_pnl"].mean()),
        "std_total": float(dist_df["total_pnl"].std()),
        "p05_total": float(dist_df["total_pnl"].quantile(0.05)),
        "p50_total": float(dist_df["total_pnl"].quantile(0.50)),
        "p95_total": float(dist_df["total_pnl"].quantile(0.95)),
        "max_abs_consistency_error": float(np.max(np.abs(dist_df["consistency_error"]))),
    }
    summary_df = pd.DataFrame([summ])

    # Save tables
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
        "hist_png": fig1_path,
        "qq_png": fig2_path,
        "sample_spot_png": fig3_path,
        "sample_optval_png": fig4_path,
        "series": {
            "S": S, "cash": cash_series, "delta": delta_series, "optval": option_val_series
        },
        "distribution_df_head": dist_df.head(10),
        "summary_df": summary_df
    }

# ---------- Convergence sweep (mean -> 0, std -> 0) ----------
def convergence_sweep(
    S0=100.0, K=101.0, r=0.05, sigma=0.20, T=1.0, M=252, n_paths=100000, seed=9,
    freqs=(1, 2, 4, 12), simple_interest=False, save_prefix="convergence"
):
    rows = []
    for n_per_day in freqs:
        steps = int(M * n_per_day)
        out = delta_hedge_run(S0, K, r, sigma, T, steps, n_paths, seed,
                              simple_interest=simple_interest,
                              save_prefix=f"{save_prefix}_{n_per_day}x")
        df = out["summary_df"]
        rows.append({
            "n_per_day": n_per_day,
            "steps": steps,
            "dt": T/steps,
            "mean_total": float(df["mean_total"].iloc[0]),
            "std_total": float(df["std_total"].iloc[0])
        })
    conv_df = pd.DataFrame(rows).sort_values("dt", ascending=False).reset_index(drop=True)

    # Save & display
    conv_path = os.path.join(BASE_DIR, f"{save_prefix}_table.csv")
    conv_df.to_csv(conv_path, index=False)
    print("Convergence Table (Mean -> 0, Std -> 0)")
    print(conv_df.head(10))

    # Plot std vs dt (log-log)
    plt.figure()
    plt.loglog(conv_df["dt"].values, conv_df["std_total"].values, marker="o", linestyle="-")
    plt.title("Convergence: Std(PnL) vs dt")
    plt.xlabel("dt")
    plt.ylabel("Std of Total PnL")
    fig_path = os.path.join(BASE_DIR, f"{save_prefix}_std_vs_dt.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()

    return {"conv_csv": conv_path, "conv_png": fig_path, "conv_df": conv_df}

# ---------- Run a baseline experiment and a convergence sweep ----------

N_PATHS = 50000

baseline = delta_hedge_run(
    S0=100.0, K=101.0, r=0.05, sigma=0.20, T=1.0, steps=252, n_paths=N_PATHS, seed=123,
    simple_interest=False, save_prefix="baseline_exp"
)

conv = convergence_sweep(
    S0=100.0, K=101.0, r=0.05, sigma=0.20, T=1.0, M=252, n_paths=N_PATHS, seed=99,
    freqs=(1, 2, 4, 12, 52, 252), simple_interest=False, save_prefix="conv_exp"
)

# Return the file paths for downloading
baseline_paths = {
    "distribution": baseline["dist_csv"],
    "summary": baseline["summary_csv"],
    "histogram": baseline["hist_png"],
    "qq_plot": baseline["qq_png"],
    "sample_spot": baseline["sample_spot_png"],
    "sample_option_value": baseline["sample_optval_png"]
}

conv_paths = {
    "convergence_table": conv["conv_csv"],
    "convergence_plot": conv["conv_png"]
}

baseline_paths, conv_paths
print('done')