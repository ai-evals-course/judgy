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
    from judgy.plotting import plot_fnr_fpr_sensitivity, run_fnr_fpr_experiment

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

    # Run the FNR/FPR sensitivity experiment
    results = run_fnr_fpr_experiment(
        true_failure_rate=true_failure_rate,  # 10% true failure rate (90% success rate)
        baseline_fpr=0,
        baseline_fnr=0,
        fnr_range=(0, 0.5),
        fpr_range=(0, 0.5),
        n_points=20,  # 20 points in each range
        n_test_positive=50,  # 100 positive test examples
        n_test_negative=50,  # 100 negative test examples
        n_unlabeled=1000,  # 1000 unlabeled examples
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
        baseline_fpr=0,
        baseline_fnr=0,
        n_test_positive=n_test_pos,
        n_test_negative=n_test_neg,
        n_unlabeled=n_unlabeled_samples,
        save_plots=True,  # Save individual plots to plots folder
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


if __name__ == "__main__":
    main()
