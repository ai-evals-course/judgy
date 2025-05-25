"""
Synthetic data generation utilities for testing LLM judge evaluation.
"""

from typing import Tuple, Union

import numpy as np
import numpy.typing as npt


def generate_test_data(
    n_positive: int,
    n_negative: int,
    true_positive_rate: float,
    true_negative_rate: float,
    random_seed: Union[int, None] = None,
) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """
    Generate synthetic test data with known judge accuracy.

    Args:
        n_positive: Number of positive examples (true label = 1).
        n_negative: Number of negative examples (true label = 0).
        true_positive_rate: Probability that judge correctly identifies positive
            examples.
        true_negative_rate: Probability that judge correctly identifies negative
            examples.
        random_seed: Random seed for reproducibility. If None, uses current
            random state.

    Returns:
        A tuple containing:
        - test_labels: Array of true labels (0s and 1s)
        - test_preds: Array of judge predictions (0s and 1s)

    Raises:
        ValueError: If rates are not between 0 and 1, or if counts are not positive.
    """
    if not 0 <= true_positive_rate <= 1:
        raise ValueError("true_positive_rate must be between 0 and 1")
    if not 0 <= true_negative_rate <= 1:
        raise ValueError("true_negative_rate must be between 0 and 1")
    if n_positive <= 0 or n_negative <= 0:
        raise ValueError("n_positive and n_negative must be positive")

    if random_seed is not None:
        np.random.seed(random_seed)

    # Create true labels
    test_labels = np.concatenate(
        [np.ones(n_positive, dtype=int), np.zeros(n_negative, dtype=int)]
    )

    # Generate judge predictions based on accuracy rates
    n_correct_positives = int(true_positive_rate * n_positive)
    n_correct_negatives = int(true_negative_rate * n_negative)

    positive_preds = np.concatenate(
        [
            np.ones(n_correct_positives, dtype=int),
            np.zeros(n_positive - n_correct_positives, dtype=int),
        ]
    )

    negative_preds = np.concatenate(
        [
            np.zeros(n_correct_negatives, dtype=int),
            np.ones(n_negative - n_correct_negatives, dtype=int),
        ]
    )

    test_preds = np.concatenate([positive_preds, negative_preds])

    return test_labels, test_preds


def generate_unlabeled_data(
    n_samples: int,
    true_pass_rate: float,
    true_positive_rate: float,
    true_negative_rate: float,
    random_seed: Union[int, None] = None,
) -> npt.NDArray[np.int_]:
    """
    Generate synthetic unlabeled data with judge predictions.

    Args:
        n_samples: Number of unlabeled samples to generate.
        true_pass_rate: True proportion of positive examples in unlabeled data.
        true_positive_rate: Judge's TPR (probability of correctly identifying
            positives).
        true_negative_rate: Judge's TNR (probability of correctly identifying
            negatives).
        random_seed: Random seed for reproducibility. If None, uses current
            random state.

    Returns:
        Array of judge predictions on unlabeled data (0s and 1s).

    Raises:
        ValueError: If rates are not between 0 and 1, or if n_samples is not positive.
    """
    if not 0 <= true_pass_rate <= 1:
        raise ValueError("true_pass_rate must be between 0 and 1")
    if not 0 <= true_positive_rate <= 1:
        raise ValueError("true_positive_rate must be between 0 and 1")
    if not 0 <= true_negative_rate <= 1:
        raise ValueError("true_negative_rate must be between 0 and 1")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate true labels for unlabeled data
    true_labels = (np.random.rand(n_samples) < true_pass_rate).astype(int)

    # Generate judge predictions based on true labels and accuracy rates
    unlabeled_preds = np.where(
        true_labels == 1,
        (np.random.rand(n_samples) < true_positive_rate).astype(int),
        (np.random.rand(n_samples) >= true_negative_rate).astype(int),
    )

    return unlabeled_preds


def run_sensitivity_experiment(
    accuracy_values: Union[list, npt.NDArray],
    fixed_accuracy: float,
    vary_tpr: bool = True,
    true_pass_rate: float = 0.9,
    n_test_positive: int = 100,
    n_test_negative: int = 100,
    n_unlabeled: int = 1000,
    bootstrap_iterations: int = 2000,
    random_seed: Union[int, None] = None,
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Run sensitivity analysis experiment varying TPR or TNR.

    Args:
        accuracy_values: Array of TPR or TNR values to test.
        fixed_accuracy: Fixed value for the other accuracy metric.
        vary_tpr: If True, vary TPR and fix TNR. If False, vary TNR and fix TPR.
        true_pass_rate: True pass rate in the unlabeled data.
        n_test_positive: Number of positive examples in test set.
        n_test_negative: Number of negative examples in test set.
        n_unlabeled: Number of unlabeled samples.
        bootstrap_iterations: Number of bootstrap iterations for CI estimation.
        random_seed: Random seed for reproducibility.

    Returns:
        A tuple containing:
        - estimates: Array of point estimates
        - lower_bounds: Array of confidence interval lower bounds
        - upper_bounds: Array of confidence interval upper bounds

    Raises:
        ValueError: If parameters are invalid.
    """
    # Import here to avoid circular imports
    from .core import estimate_success_rate

    if random_seed is not None:
        np.random.seed(random_seed)

    accuracy_values = np.asarray(accuracy_values)
    estimates = []
    lower_bounds = []
    upper_bounds = []

    for accuracy_val in accuracy_values:
        if vary_tpr:
            tpr, tnr = accuracy_val, fixed_accuracy
        else:
            tpr, tnr = fixed_accuracy, accuracy_val

        # Generate test data
        test_labels, test_preds = generate_test_data(
            n_test_positive, n_test_negative, tpr, tnr
        )

        # Generate unlabeled data
        unlabeled_preds = generate_unlabeled_data(n_unlabeled, true_pass_rate, tpr, tnr)

        # Estimate pass rate
        try:
            theta_hat, lower, upper = estimate_success_rate(
                test_labels,
                test_preds,
                unlabeled_preds,
                bootstrap_iterations=bootstrap_iterations,
            )
            estimates.append(theta_hat)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
        except (ValueError, RuntimeError):
            # Handle cases where estimation fails (e.g., poor judge performance)
            estimates.append(np.nan)
            lower_bounds.append(np.nan)
            upper_bounds.append(np.nan)

    return np.array(estimates), np.array(lower_bounds), np.array(upper_bounds)


def create_example_dataset(
    scenario: str = "good_judge",
    random_seed: Union[int, None] = 42,
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Create example datasets for different judge performance scenarios.

    Args:
        scenario: One of "good_judge", "mediocre_judge", "biased_judge", or
            "poor_judge".
        random_seed: Random seed for reproducibility.

    Returns:
        A tuple containing:
        - test_labels: Test set true labels
        - test_preds: Test set judge predictions
        - unlabeled_preds: Unlabeled set judge predictions

    Raises:
        ValueError: If scenario is not recognized.
    """
    scenarios = {
        "good_judge": {"tpr": 0.95, "tnr": 0.90, "true_rate": 0.8},
        "mediocre_judge": {"tpr": 0.75, "tnr": 0.70, "true_rate": 0.6},
        "biased_judge": {
            "tpr": 0.90,
            "tnr": 0.60,
            "true_rate": 0.7,
        },  # Better at positives
        "poor_judge": {"tpr": 0.60, "tnr": 0.55, "true_rate": 0.5},
    }

    if scenario not in scenarios:
        raise ValueError(
            f"Unknown scenario '{scenario}'. Choose from: {list(scenarios.keys())}"
        )

    params = scenarios[scenario]

    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate test data (balanced)
    test_labels, test_preds = generate_test_data(
        n_positive=50,
        n_negative=50,
        true_positive_rate=params["tpr"],
        true_negative_rate=params["tnr"],
    )

    # Generate unlabeled data
    unlabeled_preds = generate_unlabeled_data(
        n_samples=500,
        true_pass_rate=params["true_rate"],
        true_positive_rate=params["tpr"],
        true_negative_rate=params["tnr"],
    )

    return test_labels, test_preds, unlabeled_preds
