#!/usr/bin/env python3
"""
Example usage of the judgy package for TPR/TNR sensitivity analysis in the course reader.

This script generates plots that match the style shown in research papers,
with TPR/TNR sensitivity analysis showing:
- Confidence intervals as shaded blue regions
- Corrected estimates as blue x markers
- Raw observed rates as red dots
- True rate as a dashed black line
"""

import numpy as np

# Check if plotting is available
try:
    from judgy.plotting import plot_fnr_fpr_sensitivity, run_fnr_fpr_experiment, plot_label_size_sensitivity, run_label_size_experiment

    PLOTTING_AVAILABLE = True
except ImportError:
    print(
        "Plotting functionality not available. Install with: pip install judgy[plotting]"
    )
    PLOTTING_AVAILABLE = False


def main():
    """Generate research-style TPR/TNR sensitivity plots."""
    print("judgy: LLM Judge Evaluation with Bias Correction")
    print("=" * 50)

    if not PLOTTING_AVAILABLE:
        print("Error: Plotting functionality not available.")
        print("Please install with: pip install judgy[plotting]")
        return

    print("Generating TPR/TNR sensitivity plots...")
    print("This may take a few minutes due to bootstrap sampling...")

    true_pass_rate = 0.8
    true_failure_rate = 1 - true_pass_rate
    baseline_fpr = 0
    baseline_fnr = 0
    n_test_positive = 50
    n_test_negative = 50
    n_unlabeled = 1000

    # Run the FNR/FPR sensitivity experiment
    results = run_fnr_fpr_experiment(
        true_failure_rate=true_failure_rate,
        baseline_fpr=baseline_fpr,
        baseline_fnr=baseline_fnr,
        fnr_range=(0, 0.5),
        fpr_range=(0, 0.5),
        n_points=20,  # 20 points in each range
        n_test_positive=n_test_positive,
        n_test_negative=n_test_negative,
        n_unlabeled=n_unlabeled,
        # random_seed=42
    )

    (
        fnr_values,
        fnr_estimates,
        fnr_lower,
        fnr_upper,
        fnr_raw_rates,
        fpr_values,
        fpr_estimates,
        fpr_lower,
        fpr_upper,
        fpr_raw_rates,
        n_test_pos,
        n_test_neg,
        n_unlabeled_samples,
    ) = results

    # Create the research-style plots
    plot_fnr_fpr_sensitivity(
        fnr_values=fnr_values,
        fnr_estimates=fnr_estimates,
        fnr_lower=fnr_lower,
        fnr_upper=fnr_upper,
        fnr_raw_rates=fnr_raw_rates,
        fpr_values=fpr_values,
        fpr_estimates=fpr_estimates,
        fpr_lower=fpr_lower,
        fpr_upper=fpr_upper,
        fpr_raw_rates=fpr_raw_rates,
        true_rate=true_pass_rate,  # 90% true success rate
        baseline_fpr=baseline_fpr,
        baseline_fnr=baseline_fnr,
        n_test_positive=n_test_pos,
        n_test_negative=n_test_neg,
        n_unlabeled=n_unlabeled_samples,
        # save_plots=True,  # Save individual plots to plots folder
    )

    # Print experiment summary
    print(f"\nExperiment Summary:")
    print(f"True success rate: 90%")
    print(
        f"TPR range tested: {(1-fnr_values[-1])*100:.0f}% to {(1-fnr_values[0])*100:.0f}%"
    )
    print(
        f"TNR range tested: {(1-fpr_values[-1])*100:.0f}% to {(1-fpr_values[0])*100:.0f}%"
    )
    print(f"Test set size: 100 (50 positive + 50 negative)")
    print(f"Unlabeled set size: 1000")

    # Show some example results
    print(f"\nExample Results (TPR sensitivity):")
    for i in [0, len(fnr_values) // 2, -1]:
        if not np.isnan(fnr_estimates[i]):
            tpr_val = (1 - fnr_values[i]) * 100
            print(
                f"  TPR={tpr_val:.0f}%: "
                f"Raw={fnr_raw_rates[i]*100:.1f}%, "
                f"Corrected={fnr_estimates[i]*100:.1f}% "
                f"[{fnr_lower[i]*100:.1f}%, {fnr_upper[i]*100:.1f}%]"
            )

    print("\n" + "=" * 50)
    print("Example completed!")
    print("\nThe plots show:")
    print("- Blue shaded regions: 95% confidence intervals")
    print("- Blue x markers: Bias-corrected estimates")
    print("- Red dots: Raw observed rates (uncorrected)")
    print("- Black dashed line: True success rate")
    print("\nKey insights:")
    print("- Corrected estimates are closer to the true rate")
    print("- Confidence intervals quantify uncertainty")
    print("- Raw rates can be significantly biased")


def run_label_size_experiment_demo():
    """Demonstrate how estimation improves with more labeled data."""
    print("\n" + "=" * 50)
    print("LABEL SIZE SENSITIVITY ANALYSIS")
    print("=" * 50)
    print("Generating label size sensitivity plots...")
    print("This shows how estimation improves with more labeled data...")

    # Fixed parameters for this experiment
    true_pass_rate = 0.8
    true_failure_rate = 1 - true_pass_rate
    fixed_tpr = 0.9  # 90% TPR (good judge)
    fixed_tnr = 0.85  # 85% TNR (good judge)
    n_unlabeled = 1000

    # Run the label size sensitivity experiment
    results = run_label_size_experiment(
        true_failure_rate=true_failure_rate,
        fixed_tpr=fixed_tpr,
        fixed_tnr=fixed_tnr,
        label_sizes=[5, 10, 25, 50, 100, 200, 300, 400],
        n_unlabeled=n_unlabeled,
        random_seed=42,
    )

    (
        label_sizes,
        estimates,
        lower_bounds,
        upper_bounds,
        raw_rates,
        ci_widths,
    ) = results

    # Create the label size sensitivity plots
    plot_label_size_sensitivity(
        label_sizes=label_sizes,
        estimates=estimates,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        raw_rates=raw_rates,
        ci_widths=ci_widths,
        true_rate=true_pass_rate,
        fixed_tpr=fixed_tpr,
        fixed_tnr=fixed_tnr,
        n_unlabeled=n_unlabeled,
        # save_plots=True,  # Save individual plots to plots folder
    )

    # Print experiment summary
    print(f"\nLabel Size Experiment Summary:")
    print(f"True success rate: {true_pass_rate*100:.0f}%")
    print(f"Fixed TPR: {fixed_tpr*100:.0f}%")
    print(f"Fixed TNR: {fixed_tnr*100:.0f}%")
    print(f"Label sizes tested: {label_sizes[0]} to {label_sizes[-1]}")
    print(f"Unlabeled set size: {n_unlabeled}")

    # Show some example results
    print(f"\nExample Results (Label Size Impact):")
    for i in [0, len(label_sizes) // 2, -1]:
        if not np.isnan(estimates[i]):
            print(
                f"  {label_sizes[i]} labels: "
                f"Raw={raw_rates[i]*100:.1f}%, "
                f"Corrected={estimates[i]*100:.1f}% "
                f"[{lower_bounds[i]*100:.1f}%, {upper_bounds[i]*100:.1f}%], "
                f"CI width={ci_widths[i]*100:.1f}%"
            )

    print("\n" + "=" * 50)
    print("Label Size Experiment completed!")
    print("\nThe plots show:")
    print("- Left: How estimates converge to true rate with more labels")
    print("- Right: How confidence intervals get narrower with more labels")
    print("- Blue shaded regions: 95% confidence intervals")
    print("- Blue x markers: Bias-corrected estimates")
    print("- Red dots: Raw observed rates (uncorrected)")
    print("- Purple line: Confidence interval width")
    print("\nKey insights:")
    print("- More labeled data improves estimation precision")
    print("- Confidence intervals get narrower with larger label sets")
    print("- Even small label sets can provide useful bias correction")


if __name__ == "__main__":
    main()
    
    # Also run the label size experiment
    if PLOTTING_AVAILABLE:
        run_label_size_experiment_demo()
