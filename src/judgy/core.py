"""
Core functionality for estimating pass rates with LLM judge correction.
"""

from typing import Tuple, Union

import numpy as np
import numpy.typing as npt


def estimate_success_rate(
    test_labels: Union[list, npt.NDArray],
    test_preds: Union[list, npt.NDArray],
    unlabeled_preds: Union[list, npt.NDArray],
    bootstrap_iterations: int = 20000,
    confidence_level: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Estimate true pass rate and confidence interval.

    This function corrects for LLM judge bias by using labeled test data to estimate
    the judge's true positive rate (TPR) and true negative rate (TNR), then applies
    a correction to estimate the true pass rate from unlabeled predictions.
    Bootstrap resampling is used to compute confidence intervals.

    Args:
        test_labels: Array-like of 0/1 values representing human labels on test set
                    (1 = Pass, 0 = Fail).
        test_preds: Array-like of 0/1 values representing judge predictions on test set
                   (1 = Pass, 0 = Fail).
        unlabeled_preds: Array-like of 0/1 values representing judge predictions on
                        unlabeled data (1 = Pass, 0 = Fail).
        bootstrap_iterations: Number of bootstrap iterations for confidence interval
                             estimation. Default is 20000.
        confidence_level: Confidence level for the interval (between 0 and 1).
                         Default is 0.95 for 95% confidence interval.

    Returns:
        A tuple containing:
        - theta_hat: Point estimate of true pass rate (float between 0 and 1)
        - lower_bound: Lower bound of confidence interval
        - upper_bound: Upper bound of confidence interval

    Raises:
        ValueError: If judge accuracy is too low for correction (TPR + TNR <= 1)
                   or if confidence_level is not between 0 and 1.
        RuntimeError: If no valid bootstrap samples could be generated.

    Example:
        >>> test_labels = [1, 1, 0, 0, 1, 0]
        >>> test_preds = [1, 0, 0, 1, 1, 0]
        >>> unlabeled_preds = [1, 1, 0, 1, 0]
        >>> theta_hat, lower, upper = estimate_success_rate(
        ...     test_labels, test_preds, unlabeled_preds
        ... )
        >>> print(f"Estimated pass rate: {theta_hat:.3f} [{lower:.3f}, {upper:.3f}]")
    """
    # Input validation
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")

    # Convert inputs to numpy arrays
    test_labels = np.asarray(test_labels, dtype=int)
    test_preds = np.asarray(test_preds, dtype=int)
    unlabeled_preds = np.asarray(unlabeled_preds, dtype=int)

    # Validate input arrays
    if len(test_labels) != len(test_preds):
        raise ValueError(
            f"test_labels and test_preds must have the same length. "
            f"Got {len(test_labels)} and {len(test_preds)}"
        )

    if len(test_labels) == 0:
        raise ValueError("test_labels cannot be empty")

    if len(unlabeled_preds) == 0:
        raise ValueError("unlabeled_preds cannot be empty")

    # Check that inputs are binary
    for arr, name in [
        (test_labels, "test_labels"),
        (test_preds, "test_preds"),
        (unlabeled_preds, "unlabeled_preds"),
    ]:
        if not np.all(np.isin(arr, [0, 1])):
            raise ValueError(f"{name} must contain only 0 and 1 values")

    # Step 1: Calculate judge accuracy on test set
    positive_count = test_labels.sum()
    negative_count = len(test_labels) - positive_count

    if positive_count == 0 or negative_count == 0:
        raise ValueError("test_labels must contain both positive and negative examples")

    # True Positive Rate (Sensitivity)
    true_positive_rate = ((test_labels == 1) & (test_preds == 1)).sum() / positive_count

    # True Negative Rate (Specificity)
    true_negative_rate = ((test_labels == 0) & (test_preds == 0)).sum() / negative_count

    # Step 2: Calculate raw observed pass rate on unlabeled data
    observed_pass_rate = unlabeled_preds.sum() / len(unlabeled_preds)

    # Step 3: Apply correction for point estimate
    denominator = true_positive_rate + true_negative_rate - 1
    if denominator <= 0:
        raise ValueError(
            f"Judge accuracy too low for correction. TPR + TNR = "
            f"{true_positive_rate + true_negative_rate:.3f} <= 1. "
            f"Judge must be better than random."
        )

    theta_hat = (observed_pass_rate + true_negative_rate - 1) / denominator
    theta_hat = np.clip(theta_hat, 0, 1)

    # Step 4: Bootstrap confidence interval
    test_size = len(test_labels)
    indices = np.arange(test_size)
    bootstrap_samples = []

    for _ in range(bootstrap_iterations):
        # Bootstrap resample test data
        bootstrap_indices = np.random.choice(indices, size=test_size, replace=True)
        bootstrap_labels = test_labels[bootstrap_indices]
        bootstrap_preds = test_preds[bootstrap_indices]

        # Calculate counts for this bootstrap sample
        pos_count_boot = bootstrap_labels.sum()
        neg_count_boot = test_size - pos_count_boot

        # Skip if no positive or negative examples in bootstrap sample
        if pos_count_boot == 0 or neg_count_boot == 0:
            continue

        # Calculate TPR and TNR for bootstrap sample
        tpr_bootstrap = (
            (bootstrap_labels == 1) & (bootstrap_preds == 1)
        ).sum() / pos_count_boot
        tnr_bootstrap = (
            (bootstrap_labels == 0) & (bootstrap_preds == 0)
        ).sum() / neg_count_boot

        # Apply correction
        denom_bootstrap = tpr_bootstrap + tnr_bootstrap - 1
        if denom_bootstrap <= 0:
            continue

        theta_bootstrap = (observed_pass_rate + tnr_bootstrap - 1) / denom_bootstrap
        bootstrap_samples.append(np.clip(theta_bootstrap, 0, 1))

    if not bootstrap_samples:
        raise RuntimeError(
            "No valid bootstrap samples generated. This may indicate "
            "insufficient test data or very poor judge performance."
        )

    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = float(np.percentile(bootstrap_samples, lower_percentile))
    upper_bound = float(np.percentile(bootstrap_samples, upper_percentile))

    return float(theta_hat), lower_bound, upper_bound
