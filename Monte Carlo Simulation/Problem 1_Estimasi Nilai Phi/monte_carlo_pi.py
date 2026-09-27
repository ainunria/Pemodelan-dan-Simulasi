"""
Monte Carlo Simulation - Estimating the Value of Pi

Project: Monte Carlo Simulation
Topic: Estimate the Value of Pi

Requirements implemented:
1. Simulation for 10,000, 100,000, and 1,000,000 points.
2. Visualization of random points and quarter circle.
3. Accuracy and percentage error compared with true pi.
"""

import math
import numpy as np
import matplotlib.pyplot as plt


def estimate_pi(n_points, seed=42):
    """
    Estimate pi using Monte Carlo simulation.

    Random points are generated in a unit square [0,1] x [0,1].
    A point is inside the quarter circle when:
        x^2 + y^2 <= 1

    The estimated value is:
        pi ≈ 4 * (points inside quarter circle / total points)
    """
    rng = np.random.default_rng(seed)

    x = rng.random(n_points)
    y = rng.random(n_points)

    inside = (x**2 + y**2) <= 1
    pi_estimate = 4 * np.sum(inside) / n_points

    return pi_estimate, x, y, inside


def calculate_accuracy(estimate):
    """Return accuracy percentage compared with the true value of pi."""
    return 100 * (1 - abs(estimate - math.pi) / math.pi)


def calculate_error(estimate):
    """Return absolute percentage error compared with the true value of pi."""
    return abs(estimate - math.pi) / math.pi * 100


def main():
    # Number of points required by the project
    point_counts = [10_000, 100_000, 1_000_000]

    print("=" * 70)
    print("MONTE CARLO SIMULATION - ESTIMATION OF PI")
    print("=" * 70)
    print(f"{'Points':>12} {'Pi Estimate':>15} {'Error (%)':>15} {'Accuracy (%)':>17}")
    print("-" * 70)

    results = []

    for n in point_counts:
        estimate, _, _, _ = estimate_pi(n)

        error = calculate_error(estimate)
        accuracy = calculate_accuracy(estimate)

        results.append((n, estimate, error, accuracy))

        print(f"{n:>12,} {estimate:>15.8f} {error:>15.6f} {accuracy:>17.6f}")

    print("-" * 70)
    print(f"True value of pi: {math.pi:.8f}")

    # Visualize 10,000 points
    n_plot = 10_000
    estimate, x, y, inside = estimate_pi(n_plot)

    theta = np.linspace(0, np.pi / 2, 400)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    plt.figure(figsize=(7, 7))
    plt.scatter(
        x[inside], y[inside],
        s=3, alpha=0.45,
        label="Inside quarter circle"
    )
    plt.scatter(
        x[~inside], y[~inside],
        s=3, alpha=0.45,
        label="Outside quarter circle"
    )
    plt.plot(circle_x, circle_y, linewidth=2, label="Quarter circle")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Monte Carlo Estimation of π (N = {n_plot:,})")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
