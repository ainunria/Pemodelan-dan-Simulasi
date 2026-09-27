"""
Monte Carlo Simulation - Investment Portfolio Optimization

Topic:
    Monte Carlo Simulation for Investment Portfolio Optimization

Description:
    This program simulates possible annual returns of a portfolio
    consisting of multiple assets using expected returns, volatility,
    and correlations.

Assets:
    - Stock A
    - Stock B
    - Bond

The program:
    1. Defines expected returns, volatility, and correlation.
    2. Builds the covariance matrix.
    3. Generates correlated random asset returns.
    4. Simulates candidate portfolios.
    5. Evaluates return, volatility, probability of loss, and Sharpe ratio.
    6. Generates 10,000 random portfolio allocations.
    7. Visualizes the risk-return trade-off and return distribution.
"""

import math
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Asset parameters
# ============================================================

assets = ["Stock A", "Stock B", "Bond"]

# Expected annual returns
expected_returns = np.array([0.12, 0.09, 0.05])

# Annual volatility
volatilities = np.array([0.22, 0.18, 0.07])

# Correlation between assets
correlation = np.array([
    [1.00, 0.55, 0.10],
    [0.55, 1.00, 0.05],
    [0.10, 0.05, 1.00]
])

# Covariance matrix
covariance = np.outer(volatilities, volatilities) * correlation


# ============================================================
# 2. Functions
# ============================================================

def simulate_asset_returns(n, means, covariance_matrix, seed=42):
    """
    Generate correlated annual asset returns using a
    multivariate normal distribution.
    """
    rng = np.random.default_rng(seed)

    return rng.multivariate_normal(
        mean=means,
        cov=covariance_matrix,
        size=n
    )


def simulate_portfolio_returns(asset_returns, weights):
    """Calculate portfolio return for every simulation."""
    return asset_returns @ weights


def portfolio_statistics(portfolio_returns):
    """Calculate important portfolio statistics."""
    return {
        "mean": np.mean(portfolio_returns),
        "volatility": np.std(portfolio_returns, ddof=1),
        "var_5": np.percentile(portfolio_returns, 5),
        "var_95": np.percentile(portfolio_returns, 95),
        "probability_loss": np.mean(portfolio_returns < 0)
    }


def portfolio_return(weights):
    """Expected portfolio return."""
    return weights @ expected_returns


def portfolio_volatility(weights):
    """Portfolio volatility based on covariance."""
    return math.sqrt(weights @ covariance @ weights)


def sharpe_ratio(weights, risk_free_rate=0.03):
    """Simplified annual Sharpe ratio."""
    return (
        portfolio_return(weights) - risk_free_rate
    ) / portfolio_volatility(weights)


# ============================================================
# 3. Monte Carlo simulation
# ============================================================

N_SIMULATIONS = 100_000

asset_returns = simulate_asset_returns(
    N_SIMULATIONS,
    expected_returns,
    covariance,
    seed=42
)


# ============================================================
# 4. Candidate portfolios
# ============================================================

portfolios = {
    "Conservative": np.array([0.20, 0.20, 0.60]),
    "Balanced": np.array([0.40, 0.30, 0.30]),
    "Growth": np.array([0.55, 0.30, 0.15])
}


print("=" * 80)
print("MONTE CARLO SIMULATION - INVESTMENT PORTFOLIO")
print("=" * 80)

for name, weights in portfolios.items():

    simulated_returns = simulate_portfolio_returns(
        asset_returns,
        weights
    )

    stats = portfolio_statistics(simulated_returns)

    print(f"\n{name}")
    print("-" * 80)

    for asset, weight in zip(assets, weights):
        print(f"{asset:10s}: {weight:.2%}")

    print(f"Mean return       : {stats['mean']:.4%}")
    print(f"Volatility        : {stats['volatility']:.4%}")
    print(f"5th percentile    : {stats['var_5']:.4%}")
    print(f"95th percentile   : {stats['var_95']:.4%}")
    print(f"Probability loss  : {stats['probability_loss']:.4%}")
    print(f"Sharpe ratio      : {sharpe_ratio(weights):.4f}")


# ============================================================
# 5. Random portfolio search
# ============================================================

rng = np.random.default_rng(42)

N_RANDOM_PORTFOLIOS = 10_000

random_weights = rng.dirichlet(
    np.ones(len(assets)),
    size=N_RANDOM_PORTFOLIOS
)

random_returns = random_weights @ expected_returns

random_volatility = np.sqrt(
    np.einsum(
        "ij,jk,ik->i",
        random_weights,
        covariance,
        random_weights
    )
)

random_sharpe = (
    random_returns - 0.03
) / random_volatility

min_vol_idx = np.argmin(random_volatility)
max_sharpe_idx = np.argmax(random_sharpe)


print("\n" + "=" * 80)
print("RANDOM PORTFOLIO SEARCH")
print("=" * 80)

print("\nMinimum-volatility candidate:")
for asset, weight in zip(
    assets,
    random_weights[min_vol_idx]
):
    print(f"{asset:10s}: {weight:.2%}")

print(f"Expected return: {random_returns[min_vol_idx]:.4%}")
print(f"Volatility    : {random_volatility[min_vol_idx]:.4%}")

print("\nMaximum-Sharpe candidate:")
for asset, weight in zip(
    assets,
    random_weights[max_sharpe_idx]
):
    print(f"{asset:10s}: {weight:.2%}")

print(f"Expected return: {random_returns[max_sharpe_idx]:.4%}")
print(f"Volatility    : {random_volatility[max_sharpe_idx]:.4%}")
print(f"Sharpe ratio  : {random_sharpe[max_sharpe_idx]:.4f}")


# ============================================================
# 6. Visualization - Balanced portfolio
# ============================================================

balanced_returns = simulate_portfolio_returns(
    asset_returns,
    portfolios["Balanced"]
)

plt.figure(figsize=(8, 5))
plt.hist(
    balanced_returns,
    bins=100,
    density=True,
    alpha=0.75
)

plt.axvline(
    np.mean(balanced_returns),
    linestyle="--",
    linewidth=2,
    label=f"Mean = {np.mean(balanced_returns):.2%}"
)

plt.axvline(
    0,
    linestyle=":",
    linewidth=2,
    label="Break-even"
)

plt.xlabel("Annual Portfolio Return")
plt.ylabel("Density")
plt.title("Monte Carlo Distribution - Balanced Portfolio")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()


# ============================================================
# 7. Visualization - Risk-return
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    random_volatility,
    random_returns,
    s=5,
    alpha=0.25,
    label="Random portfolios"
)

plt.scatter(
    random_volatility[max_sharpe_idx],
    random_returns[max_sharpe_idx],
    s=80,
    marker="*",
    label="Maximum-Sharpe candidate"
)

plt.scatter(
    random_volatility[min_vol_idx],
    random_returns[min_vol_idx],
    s=80,
    marker="X",
    label="Minimum-volatility candidate"
)

plt.xlabel("Annual Volatility (Risk)")
plt.ylabel("Expected Annual Return")
plt.title("Monte Carlo Risk-Return Trade-off")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
