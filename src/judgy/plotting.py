"""
Plotting utilities for visualizing LLM judge evaluation results.
"""

import os
from typing import List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def _check_matplotlib() -> None:
    """Check if matplotlib is available and raise informative error if not."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for plotting functionality. "
            "Install it with: pip install judgy[plotting]"
        )


def plot_fnr_fpr_sensitivity(
    fnr_values: Union[List[float], np.ndarray],
    fnr_estimates: Union[List[float], np.ndarray],
    fnr_lower: Union[List[float], np.ndarray],
    fnr_upper: Union[List[float], np.ndarray],
    fnr_raw_rates: Union[List[float], np.ndarray],
    fpr_values: Union[List[float], np.ndarray],
    fpr_estimates: Union[List[float], np.ndarray],
    fpr_lower: Union[List[float], np.ndarray],
    fpr_upper: Union[List[float], np.ndarray],
    fpr_raw_rates: Union[List[float], np.ndarray],
    true_rate: float,
    baseline_fpr: float = 0.05,
    baseline_fnr: float = 0.05,
    n_test_positive: int = 100,
    n_test_negative: int = 100,
    n_unlabeled: int = 1000,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
    save_plots: bool = False,
) -> None:
    """
    Create TPR/TNR sensitivity plots matching the research paper style.

    Args:
        fnr_values: Array of FNR values (False Negative Rate = 1 - TPR).
        fnr_estimates: Corrected estimates for FNR experiment.
        fnr_lower: Lower bounds for FNR experiment.
        fnr_upper: Upper bounds for FNR experiment.
        fnr_raw_rates: Raw observed rates for FNR experiment.
        fpr_values: Array of FPR values (False Positive Rate = 1 - TNR).
        fpr_estimates: Corrected estimates for FPR experiment.
        fpr_lower: Lower bounds for FPR experiment.
        fpr_upper: Upper bounds for FPR experiment.
        fpr_raw_rates: Raw observed rates for FPR experiment.
        true_rate: The true success rate to show as reference line.
        baseline_fpr: Fixed FPR value used in FNR sensitivity experiment.
        baseline_fnr: Fixed FNR value used in FPR sensitivity experiment.
        n_test_positive: Number of positive examples in test set.
        n_test_negative: Number of negative examples in test set.
        n_unlabeled: Number of unlabeled samples.
        figsize: Figure size as (width, height) tuple.
        save_path: If provided, saves the plot to this path.
        save_plots: If True, saves each subplot individually to a plots folder.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    _check_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Convert FNR/FPR to TPR/TNR and convert failure rates to success rates
    tpr_values = (1 - np.asarray(fnr_values)) * 100  # TPR = 1 - FNR
    tpr_estimates = np.asarray(fnr_estimates) * 100  # Already success rates
    tpr_lower = np.asarray(fnr_lower) * 100
    tpr_upper = np.asarray(fnr_upper) * 100
    tpr_raw_rates = np.asarray(fnr_raw_rates) * 100

    tnr_values = (1 - np.asarray(fpr_values)) * 100  # TNR = 1 - FPR
    tnr_estimates = np.asarray(fpr_estimates) * 100  # Already success rates
    tnr_lower = np.asarray(fpr_lower) * 100
    tnr_upper = np.asarray(fpr_upper) * 100
    tnr_raw_rates = np.asarray(fpr_raw_rates) * 100

    true_rate_pct = true_rate * 100
    baseline_tpr = (1 - baseline_fnr) * 100
    baseline_tnr = (1 - baseline_fpr) * 100

    # Left plot: CI vs. TPR
    axes[0].fill_between(
        tpr_values, tpr_lower, tpr_upper, alpha=0.3, color="lightblue", label="95% CI"
    )
    axes[0].plot(
        tpr_values,
        tpr_estimates,
        "x-",
        color="blue",
        markersize=8,
        linewidth=2,
        label="Corrected θ̂",
    )
    axes[0].plot(
        tpr_values, tpr_raw_rates, "o", color="red", markersize=6, label="Raw k/m"
    )
    axes[0].axhline(
        true_rate_pct,
        linestyle="--",
        color="black",
        linewidth=1.5,
        label=f"True θ = {true_rate_pct:.0f}%",
    )

    axes[0].set_xlabel("TPR (%)", fontsize=12)
    axes[0].set_ylabel("Estimated success rate θ̂ (%)", fontsize=12)
    axes[0].set_title(f"CI vs. TPR (baseline TNR={baseline_tnr:.0f}%)", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_ylim(0, 100)

    # Right plot: CI vs. TNR
    axes[1].fill_between(
        tnr_values, tnr_lower, tnr_upper, alpha=0.3, color="lightblue", label="95% CI"
    )
    axes[1].plot(
        tnr_values,
        tnr_estimates,
        "x-",
        color="blue",
        markersize=8,
        linewidth=2,
        label="Corrected θ̂",
    )
    axes[1].plot(
        tnr_values, tnr_raw_rates, "o", color="red", markersize=6, label="Raw k/m"
    )
    axes[1].axhline(
        true_rate_pct,
        linestyle="--",
        color="black",
        linewidth=1.5,
        label=f"True θ = {true_rate_pct:.0f}%",
    )

    axes[1].set_xlabel("TNR (%)", fontsize=12)
    axes[1].set_ylabel("Estimated success rate θ̂ (%)", fontsize=12)
    axes[1].set_title(f"CI vs. TNR (baseline TPR={baseline_tpr:.0f}%)", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_ylim(0, 100)

    # Add sample size annotation
    axes[0].text(
        0.02,
        0.02,
        f"{n_test_negative} human-provided fail labels, "
        f"{n_test_positive} human-provided pass labels\n"
        f"m = {n_unlabeled} examples",
        transform=axes[0].transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    axes[1].text(
        0.02,
        0.02,
        f"{n_test_negative} human-provided fail labels, "
        f"{n_test_positive} human-provided pass labels\n"
        f"m = {n_unlabeled} examples",
        transform=axes[1].transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if save_plots:
        # Create plots directory if it doesn't exist
        os.makedirs("plots", exist_ok=True)

        # Save each subplot individually
        # Save left plot (TPR sensitivity)
        fig_left, ax_left = plt.subplots(figsize=(6, 5))
        ax_left.fill_between(
            tpr_values,
            tpr_lower,
            tpr_upper,
            alpha=0.3,
            color="lightblue",
            label="95% CI",
        )
        ax_left.plot(
            tpr_values,
            tpr_estimates,
            "x-",
            color="blue",
            markersize=8,
            linewidth=2,
            label="Corrected θ̂",
        )
        ax_left.plot(
            tpr_values, tpr_raw_rates, "o", color="red", markersize=6, label="Raw k/m"
        )
        ax_left.axhline(
            true_rate_pct,
            linestyle="--",
            color="black",
            linewidth=1.5,
            label=f"True θ = {true_rate_pct:.0f}%",
        )
        ax_left.set_xlabel("TPR (%)", fontsize=12)
        ax_left.set_ylabel("Estimated success rate θ̂ (%)", fontsize=12)
        ax_left.set_title(f"CI vs. TPR (baseline TNR={baseline_tnr:.0f}%)", fontsize=12)
        ax_left.grid(True, alpha=0.3)
        ax_left.legend()
        ax_left.set_ylim(0, 100)
        ax_left.text(
            0.02,
            0.02,
            f"{n_test_negative} human-provided fail labels, "
            f"{n_test_positive} human-provided pass labels\n"
            f"m = {n_unlabeled} examples",
            transform=ax_left.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        plt.tight_layout()
        plt.savefig("plots/tpr_sensitivity.png", dpi=300, bbox_inches="tight")
        plt.close(fig_left)

        # Save right plot (TNR sensitivity)
        fig_right, ax_right = plt.subplots(figsize=(6, 5))
        ax_right.fill_between(
            tnr_values,
            tnr_lower,
            tnr_upper,
            alpha=0.3,
            color="lightblue",
            label="95% CI",
        )
        ax_right.plot(
            tnr_values,
            tnr_estimates,
            "x-",
            color="blue",
            markersize=8,
            linewidth=2,
            label="Corrected θ̂",
        )
        ax_right.plot(
            tnr_values, tnr_raw_rates, "o", color="red", markersize=6, label="Raw k/m"
        )
        ax_right.axhline(
            true_rate_pct,
            linestyle="--",
            color="black",
            linewidth=1.5,
            label=f"True θ = {true_rate_pct:.0f}%",
        )
        ax_right.set_xlabel("TNR (%)", fontsize=12)
        ax_right.set_ylabel("Estimated success rate θ̂ (%)", fontsize=12)
        ax_right.set_title(
            f"CI vs. TNR (baseline TPR={baseline_tpr:.0f}%)", fontsize=12
        )
        ax_right.grid(True, alpha=0.3)
        ax_right.legend()
        ax_right.set_ylim(0, 100)
        ax_right.text(
            0.02,
            0.02,
            f"{n_test_negative} human-provided fail labels, "
            f"{n_test_positive} human-provided pass labels\n"
            f"m = {n_unlabeled} examples",
            transform=ax_right.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        plt.tight_layout()
        plt.savefig("plots/tnr_sensitivity.png", dpi=300, bbox_inches="tight")
        plt.close(fig_right)

        print(
            "Individual plots saved to plots/tpr_sensitivity.png and "
            "plots/tnr_sensitivity.png"
        )

    plt.show()


def run_fnr_fpr_experiment(
    true_failure_rate: float = 0.1,
    baseline_fpr: float = 0.05,
    baseline_fnr: float = 0.05,
    fnr_range: Tuple[float, float] = (0.0, 0.9),
    fpr_range: Tuple[float, float] = (0.0, 0.9),
    n_points: int = 10,
    n_test_positive: int = 100,
    n_test_negative: int = 100,
    n_unlabeled: int = 1000,
    bootstrap_iterations: int = 2000,
    random_seed: Union[int, None] = 42,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
    int,
]:
    """
    Run FNR/FPR sensitivity experiment to generate data for the research-style plots.

    Args:
        true_failure_rate: True failure rate in the unlabeled data.
        baseline_fpr: Fixed FPR value for FNR sensitivity experiment.
        baseline_fnr: Fixed FNR value used in FPR sensitivity experiment.
        fnr_range: (min, max) range for FNR values.
        fpr_range: (min, max) range for FPR values.
        n_points: Number of points to test in each range.
        n_test_positive: Number of positive examples in test set.
        n_test_negative: Number of negative examples in test set.
        n_unlabeled: Number of unlabeled samples.
        bootstrap_iterations: Number of bootstrap iterations.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (fnr_values, fnr_estimates, fnr_lower, fnr_upper, fnr_raw_rates,
                 fpr_values, fpr_estimates, fpr_lower, fpr_upper, fpr_raw_rates,
                 n_test_positive, n_test_negative, n_unlabeled)
    """
    from .core import estimate_success_rate
    from .synthetic import generate_test_data, generate_unlabeled_data

    if random_seed is not None:
        np.random.seed(random_seed)

    true_success_rate = 1 - true_failure_rate

    # FNR sensitivity (varying FNR, fixed FPR)
    fnr_values = np.linspace(fnr_range[0], fnr_range[1], n_points)
    fnr_estimates, fnr_lower, fnr_upper, fnr_raw_rates = [], [], [], []

    for fnr in fnr_values:
        tpr = 1 - fnr  # True Positive Rate = 1 - False Negative Rate
        tnr = 1 - baseline_fpr  # True Negative Rate = 1 - False Positive Rate

        # Generate test data with fixed seed for reproducibility
        test_labels, test_preds = generate_test_data(
            n_test_positive, n_test_negative, tpr, tnr, random_seed=42
        )

        # Generate unlabeled data with fixed seed for reproducibility
        unlabeled_preds = generate_unlabeled_data(
            n_unlabeled, true_success_rate, tpr, tnr, random_seed=42
        )

        # Calculate raw observed success rate
        raw_success_rate = unlabeled_preds.sum() / len(unlabeled_preds)
        fnr_raw_rates.append(raw_success_rate)

        try:
            # Estimate success rate directly
            theta_hat, lower, upper = estimate_success_rate(
                test_labels,
                test_preds,
                unlabeled_preds,
                bootstrap_iterations=bootstrap_iterations,
            )
            fnr_estimates.append(theta_hat)
            fnr_lower.append(lower)
            fnr_upper.append(upper)
        except (ValueError, RuntimeError):
            fnr_estimates.append(np.nan)
            fnr_lower.append(np.nan)
            fnr_upper.append(np.nan)

    # FPR sensitivity (varying FPR, fixed FNR)
    fpr_values = np.linspace(fpr_range[0], fpr_range[1], n_points)
    fpr_estimates, fpr_lower, fpr_upper, fpr_raw_rates = [], [], [], []

    for fpr in fpr_values:
        tpr = 1 - baseline_fnr  # True Positive Rate = 1 - False Negative Rate
        tnr = 1 - fpr  # True Negative Rate = 1 - False Positive Rate

        # Generate test data with fixed seed for reproducibility
        test_labels, test_preds = generate_test_data(
            n_test_positive, n_test_negative, tpr, tnr, random_seed=42
        )

        # Generate unlabeled data with fixed seed for reproducibility
        unlabeled_preds = generate_unlabeled_data(
            n_unlabeled, true_success_rate, tpr, tnr, random_seed=42
        )

        # Calculate raw observed success rate
        raw_success_rate = unlabeled_preds.sum() / len(unlabeled_preds)
        fpr_raw_rates.append(raw_success_rate)

        try:
            # Estimate success rate directly
            theta_hat, lower, upper = estimate_success_rate(
                test_labels,
                test_preds,
                unlabeled_preds,
                bootstrap_iterations=bootstrap_iterations,
            )
            fpr_estimates.append(theta_hat)
            fpr_lower.append(lower)
            fpr_upper.append(upper)
        except (ValueError, RuntimeError):
            fpr_estimates.append(np.nan)
            fpr_lower.append(np.nan)
            fpr_upper.append(np.nan)

    return (
        np.array(fnr_values),
        np.array(fnr_estimates),
        np.array(fnr_lower),
        np.array(fnr_upper),
        np.array(fnr_raw_rates),
        np.array(fpr_values),
        np.array(fpr_estimates),
        np.array(fpr_lower),
        np.array(fpr_upper),
        np.array(fpr_raw_rates),
        n_test_positive,
        n_test_negative,
        n_unlabeled,
    )
