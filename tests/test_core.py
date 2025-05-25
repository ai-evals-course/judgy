"""
Tests for the core functionality of judgy.
"""

import numpy as np
import pytest

from judgy.core import estimate_success_rate


class TestEstimatePassRate:
    """Test cases for the estimate_success_rate function."""

    def test_basic_functionality(self):
        """Test basic functionality with known inputs."""
        # Simple test case
        test_labels = [1, 1, 0, 0, 1, 0, 1, 0]
        test_preds = [1, 0, 0, 1, 1, 0, 1, 0]
        unlabeled_preds = [1, 1, 0, 1, 0, 1, 0, 1, 1, 0]

        theta_hat, lower, upper = estimate_success_rate(
            test_labels, test_preds, unlabeled_preds, bootstrap_iterations=100
        )

        # Basic sanity checks
        assert 0 <= theta_hat <= 1
        assert 0 <= lower <= 1
        assert 0 <= upper <= 1
        assert lower <= theta_hat <= upper

    def test_perfect_judge(self):
        """Test with a perfect judge (TPR=1, TNR=1)."""
        # Perfect judge should give exact pass rate
        test_labels = [1, 1, 1, 0, 0, 0]
        test_preds = [1, 1, 1, 0, 0, 0]  # Perfect predictions
        unlabeled_preds = [1, 1, 0, 0]  # 50% pass rate

        theta_hat, lower, upper = estimate_success_rate(
            test_labels, test_preds, unlabeled_preds, bootstrap_iterations=100
        )

        # With perfect judge, estimate should be close to observed rate
        observed_rate = np.mean(unlabeled_preds)
        assert abs(theta_hat - observed_rate) < 0.1

    def test_input_validation(self):
        """Test input validation and error handling."""
        valid_labels = [1, 0, 1, 0]
        valid_preds = [1, 0, 1, 1]
        valid_unlabeled = [1, 0, 1]

        # Test mismatched lengths
        with pytest.raises(ValueError, match="same length"):
            estimate_success_rate([1, 0], [1, 0, 1], valid_unlabeled)

        # Test empty inputs
        with pytest.raises(ValueError, match="cannot be empty"):
            estimate_success_rate([], [], valid_unlabeled)

        with pytest.raises(ValueError, match="cannot be empty"):
            estimate_success_rate(valid_labels, valid_preds, [])

        # Test non-binary inputs
        with pytest.raises(ValueError, match="must contain only 0 and 1"):
            estimate_success_rate([1, 2, 0, 1], valid_preds, valid_unlabeled)

        # Test invalid confidence level
        with pytest.raises(
            ValueError, match="confidence_level must be between 0 and 1"
        ):
            estimate_success_rate(
                valid_labels, valid_preds, valid_unlabeled, confidence_level=1.5
            )

        with pytest.raises(
            ValueError, match="confidence_level must be between 0 and 1"
        ):
            estimate_success_rate(
                valid_labels, valid_preds, valid_unlabeled, confidence_level=0
            )

        # Test invalid bootstrap iterations
        with pytest.raises(ValueError, match="bootstrap_iterations must be positive"):
            estimate_success_rate(
                valid_labels, valid_preds, valid_unlabeled, bootstrap_iterations=0
            )

    def test_no_positive_or_negative_examples(self):
        """Test error handling when test set lacks positive or negative examples."""
        # All positive labels
        with pytest.raises(ValueError, match="both positive and negative examples"):
            estimate_success_rate([1, 1, 1], [1, 0, 1], [1, 0, 1])

        # All negative labels
        with pytest.raises(ValueError, match="both positive and negative examples"):
            estimate_success_rate([0, 0, 0], [1, 0, 1], [1, 0, 1])

    def test_poor_judge_accuracy(self):
        """Test error handling when judge accuracy is too low."""
        # Judge that's worse than random (TPR + TNR <= 1)
        test_labels = [1, 1, 1, 1, 0, 0, 0, 0]
        test_preds = [0, 0, 0, 0, 1, 1, 1, 1]  # Always wrong
        unlabeled_preds = [1, 0, 1, 0]

        with pytest.raises(ValueError, match="Judge accuracy too low"):
            estimate_success_rate(test_labels, test_preds, unlabeled_preds)

    def test_different_confidence_levels(self):
        """Test different confidence levels."""
        test_labels = [1, 1, 0, 0, 1, 0, 1, 0]
        test_preds = [1, 0, 0, 1, 1, 0, 1, 0]
        unlabeled_preds = [1, 1, 0, 1, 0] * 10  # Larger sample

        # Test 90% confidence interval
        theta_90, lower_90, upper_90 = estimate_success_rate(
            test_labels,
            test_preds,
            unlabeled_preds,
            confidence_level=0.90,
            bootstrap_iterations=200,
        )

        # Test 99% confidence interval
        theta_99, lower_99, upper_99 = estimate_success_rate(
            test_labels,
            test_preds,
            unlabeled_preds,
            confidence_level=0.99,
            bootstrap_iterations=200,
        )

        # Point estimates should be the same
        assert abs(theta_90 - theta_99) < 1e-10

        # 99% CI should be wider than 90% CI
        width_90 = upper_90 - lower_90
        width_99 = upper_99 - lower_99
        assert width_99 >= width_90

    def test_numpy_array_inputs(self):
        """Test that numpy arrays work as inputs."""
        test_labels = np.array([1, 1, 0, 0, 1, 0])
        test_preds = np.array([1, 0, 0, 1, 1, 0])
        unlabeled_preds = np.array([1, 1, 0, 1, 0])

        theta_hat, lower, upper = estimate_success_rate(
            test_labels, test_preds, unlabeled_preds, bootstrap_iterations=100
        )

        assert 0 <= theta_hat <= 1
        assert lower <= theta_hat <= upper

    def test_reproducibility(self):
        """Test that results are reproducible with same random seed."""
        test_labels = [1, 1, 0, 0, 1, 0, 1, 0]
        test_preds = [1, 0, 0, 1, 1, 0, 1, 0]
        unlabeled_preds = [1, 1, 0, 1, 0] * 5

        # Set seed before each call
        np.random.seed(42)
        result1 = estimate_success_rate(
            test_labels, test_preds, unlabeled_preds, bootstrap_iterations=100
        )

        np.random.seed(42)
        result2 = estimate_success_rate(
            test_labels, test_preds, unlabeled_preds, bootstrap_iterations=100
        )

        # Results should be identical
        assert result1[0] == result2[0]  # theta_hat
        assert result1[1] == result2[1]  # lower
        assert result1[2] == result2[2]  # upper

    def test_edge_case_small_sample(self):
        """Test behavior with very small samples."""
        # Minimal valid input
        test_labels = [1, 0]
        test_preds = [1, 0]  # Perfect judge
        unlabeled_preds = [1]

        theta_hat, lower, upper = estimate_success_rate(
            test_labels, test_preds, unlabeled_preds, bootstrap_iterations=50
        )

        # Should still produce valid results
        assert 0 <= theta_hat <= 1
        assert lower <= upper

    def test_large_sample_stability(self):
        """Test that larger samples give more stable results."""
        np.random.seed(42)

        # Generate larger synthetic dataset
        n_pos, n_neg = 200, 200
        tpr, tnr = 0.8, 0.85

        # Test data
        test_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
        test_preds = np.concatenate(
            [
                (np.random.rand(n_pos) < tpr).astype(int),
                (np.random.rand(n_neg) >= tnr).astype(int),
            ]
        )

        # Unlabeled data
        true_rate = 0.7
        n_unlabeled = 1000
        true_unlabeled = (np.random.rand(n_unlabeled) < true_rate).astype(int)
        unlabeled_preds = np.where(
            true_unlabeled == 1,
            (np.random.rand(n_unlabeled) < tpr).astype(int),
            (np.random.rand(n_unlabeled) >= tnr).astype(int),
        )

        theta_hat, lower, upper = estimate_success_rate(
            test_labels, test_preds, unlabeled_preds, bootstrap_iterations=500
        )

        # With large sample, estimate should be reasonably close to true rate
        assert abs(theta_hat - true_rate) < 0.1

        # Confidence interval should be reasonably narrow
        assert (upper - lower) < 0.2
