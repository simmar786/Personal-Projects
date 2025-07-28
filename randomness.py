# Exploratory Framework: Investigating Randomness in American Monte Carlo Option Pricing

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.optimize import minimize_scalar
import os
import scipy.stats as si

output_dir = os.path.join(os.path.dirname(__file__), "randomness")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 1. Randomness in Path Generation: Pseudo vs Sobol vs Noisy Monte Carlo ---
def simulate_paths(S0, r, sigma, T, N, M, method="pseudo"):
    dt = T / N
    if method == "pseudo":
        Z = np.random.normal(0, 1, (M, N))
    elif method == "sobol":
        sampler = qmc.Sobol(d=N, scramble=True)
        U = sampler.random(n=M)
        Z = si.norm.ppf(U)  # Inverse CDF to get normal samples
    else:
        raise ValueError("Unknown method")

    paths = np.zeros((M, N + 1))
    paths[:, 0] = S0
    for t in range(1, N + 1):
        paths[:, t] = paths[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])
    return paths

# --- 2. Randomness in Optimal Stopping: Add Noise to Continuation Value ---
def american_option_price_with_noise(paths, K, r, dt, noise_level=0.0):
    M, N_plus_1 = paths.shape
    N = N_plus_1 - 1
    cashflows = np.maximum(K - paths[:, -1], 0)  # European payoff

    for t in range(N - 1, 0, -1):
        in_the_money = paths[:, t] < K
        X = paths[in_the_money, t]
        Y = cashflows[in_the_money] * np.exp(-r * dt)
        if len(X) == 0:
            continue

        coeffs = np.polyfit(X, Y, deg=2)
        continuation = np.polyval(coeffs, X) + noise_level * np.random.normal(0, 1, len(X))

        exercise = np.maximum(K - X, 0)
        decision = exercise > continuation
        idx = np.where(in_the_money)[0]
        cashflows[idx[decision]] = exercise[decision]
        cashflows[idx[~decision]] = cashflows[idx[~decision]] * np.exp(-r * dt)

    return np.mean(cashflows * np.exp(-r * dt))

# --- 3. Entropy Calculation for Pathwise Payoff Distribution ---
def path_entropy(payoffs, bins=50):
    hist, bin_edges = np.histogram(payoffs, bins=bins, density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log(hist))
    return entropy

# --- Example Parameters ---
S0 = 100
K = 100
r = 0.05
sigma = 0.2
T = 1.0
N = 50
M = 10000

# Simulate paths and compute American option price under different randomness sources
pseudo_paths = simulate_paths(S0, r, sigma, T, N, M, method="pseudo")
sobol_paths = simulate_paths(S0, r, sigma, T, N, M, method="sobol")

pseudo_price = american_option_price_with_noise(pseudo_paths, K, r, T / N, noise_level=0.0)
sobol_price = american_option_price_with_noise(sobol_paths, K, r, T / N, noise_level=0.0)
noisy_price = american_option_price_with_noise(pseudo_paths, K, r, T / N, noise_level=0.5)

# Calculate entropy of final payoffs
pseudo_entropy = path_entropy(np.maximum(K - pseudo_paths[:, -1], 0))
sobol_entropy = path_entropy(np.maximum(K - sobol_paths[:, -1], 0))

# --- Output ---
print("\n--- American Option Prices ---")
print(f"Pseudo-random Monte Carlo: {pseudo_price:.4f}")
print(f"Sobol Quasi-random Monte Carlo: {sobol_price:.4f}")
print(f"With Noisy Continuation Values: {noisy_price:.4f}")

print("\n--- Entropy of Payoff Distributions ---")
print(f"Entropy (Pseudo): {pseudo_entropy:.4f}")
print(f"Entropy (Sobol): {sobol_entropy:.4f}")

# Plot payoff distribution
plt.hist(np.maximum(K - pseudo_paths[:, -1], 0), bins=50, alpha=0.5, label="Pseudo")
plt.hist(np.maximum(K - sobol_paths[:, -1], 0), bins=50, alpha=0.5, label="Sobol")
plt.title("Distribution of Terminal Payoffs")
plt.xlabel("Payoff")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plot_path = os.path.join(output_dir, "Distribution of Terminal Payoffs.png")
plt.savefig(plot_path)
plt.close()
print('End')