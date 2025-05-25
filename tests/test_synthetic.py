"""
Tests for synthetic data generation utilities.
"""

import numpy as np
import pytest

from judgy.synthetic import (
    create_example_dataset,
    generate_test_data,
    generate_unlabeled_data,
    run_sensitivity_experiment,
)


class TestGenerateTestData:
    """Test cases for generate_test_data function."""

    def test_basic_functionality(self):
        """Test basic data generation."""
        test_labels, test_preds = generate_test_data(
            n_positive=10,
            n_negative=5,
            true_positive_rate=0.8,
            true_negative_rate=0.9,
            random_seed=42,
        )

        # Check shapes and types
        assert len(test_labels) == 15
        assert len(test_preds) == 15
        assert test_labels.dtype == int
        assert test_preds.dtype == int

        # Check label structure
        assert np.sum(test_labels) == 10  # 10 positive labels
        assert np.sum(test_labels == 0) == 5  # 5 negative labels

        # Check that all values are binary
        assert np.all(np.isin(test_labels, [0, 1]))
        assert np.all(np.isin(test_preds, [0, 1]))

    def test_perfect_accuracy(self):
        """Test with perfect judge accuracy."""
        test_labels, test_preds = generate_test_data(
            n_positive=20,
            n_negative=20,
            true_positive_rate=1.0,
            true_negative_rate=1.0,
            random_seed=42,
        )

        # With perfect accuracy, predictions should match labels
        assert np.array_equal(test_labels, test_preds)

    def test_zero_accuracy(self):
        """Test with zero accuracy (always wrong)."""
        test_labels, test_preds = generate_test_data(
            n_positive=10,
            n_negative=10,
            true_positive_rate=0.0,
            true_negative_rate=0.0,
            random_seed=42,
        )

        # With zero accuracy, predictions should be opposite of labels
        expected_preds = 1 - test_labels
        assert np.array_equal(test_preds, expected_preds)

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        result1 = generate_test_data(
            n_positive=5,
            n_negative=5,
            true_positive_rate=0.7,
            true_negative_rate=0.8,
            random_seed=123,
        )

        result2 = generate_test_data(
            n_positive=5,
            n_negative=5,
            true_positive_rate=0.7,
            true_negative_rate=0.8,
            random_seed=123,
        )

        assert np.array_equal(result1[0], result2[0])
        assert np.array_equal(result1[1], result2[1])

    def test_input_validation(self):
        """Test input validation."""
        # Invalid rates
        with pytest.raises(
            ValueError, match="true_positive_rate must be between 0 and 1"
        ):
            generate_test_data(5, 5, -0.1, 0.8)

        with pytest.raises(
            ValueError, match="true_positive_rate must be between 0 and 1"
        ):
            generate_test_data(5, 5, 1.1, 0.8)

        with pytest.raises(
            ValueError, match="true_negative_rate must be between 0 and 1"
        ):
            generate_test_data(5, 5, 0.8, -0.1)

        with pytest.raises(
            ValueError, match="true_negative_rate must be between 0 and 1"
        ):
            generate_test_data(5, 5, 0.8, 1.1)

        # Invalid counts
        with pytest.raises(
            ValueError, match="n_positive and n_negative must be positive"
        ):
            generate_test_data(0, 5, 0.8, 0.8)

        with pytest.raises(
            ValueError, match="n_positive and n_negative must be positive"
        ):
            generate_test_data(5, 0, 0.8, 0.8)


class TestGenerateUnlabeledData:
    """Test cases for generate_unlabeled_data function."""

    def test_basic_functionality(self):
        """Test basic unlabeled data generation."""
        unlabeled_preds = generate_unlabeled_data(
            n_samples=100,
            true_pass_rate=0.6,
            true_positive_rate=0.8,
            true_negative_rate=0.9,
            random_seed=42,
        )

        # Check shape and type
        assert len(unlabeled_preds) == 100
        assert unlabeled_preds.dtype == int
        assert np.all(np.isin(unlabeled_preds, [0, 1]))

    def test_extreme_pass_rates(self):
        """Test with extreme pass rates."""
        # All positive
        unlabeled_preds = generate_unlabeled_data(
            n_samples=50,
            true_pass_rate=1.0,
            true_positive_rate=1.0,
            true_negative_rate=1.0,
            random_seed=42,
        )
        assert np.all(unlabeled_preds == 1)

        # All negative
        unlabeled_preds = generate_unlabeled_data(
            n_samples=50,
            true_pass_rate=0.0,
            true_positive_rate=1.0,
            true_negative_rate=1.0,
            random_seed=42,
        )
        assert np.all(unlabeled_preds == 0)

    def test_input_validation(self):
        """Test input validation."""
        # Invalid pass rate
        with pytest.raises(ValueError, match="true_pass_rate must be between 0 and 1"):
            generate_unlabeled_data(10, -0.1, 0.8, 0.8)

        with pytest.raises(ValueError, match="true_pass_rate must be between 0 and 1"):
            generate_unlabeled_data(10, 1.1, 0.8, 0.8)

        # Invalid accuracy rates
        with pytest.raises(
            ValueError, match="true_positive_rate must be between 0 and 1"
        ):
            generate_unlabeled_data(10, 0.5, -0.1, 0.8)

        with pytest.raises(
            ValueError, match="true_negative_rate must be between 0 and 1"
        ):
            generate_unlabeled_data(10, 0.5, 0.8, 1.1)

        # Invalid sample count
        with pytest.raises(ValueError, match="n_samples must be positive"):
            generate_unlabeled_data(0, 0.5, 0.8, 0.8)


class TestRunSensitivityExperiment:
    """Test cases for run_sensitivity_experiment function."""

    def test_tpr_sensitivity(self):
        """Test TPR sensitivity experiment."""
        tpr_values = [0.6, 0.8, 1.0]
        estimates, lower_bounds, upper_bounds = run_sensitivity_experiment(
            accuracy_values=tpr_values,
            fixed_accuracy=0.9,  # Fixed TNR
            vary_tpr=True,
            bootstrap_iterations=100,
            random_seed=42,
        )

        # Check output shapes
        assert len(estimates) == len(tpr_values)
        assert len(lower_bounds) == len(tpr_values)
        assert len(upper_bounds) == len(tpr_values)

        # Check that estimates are valid probabilities
        assert np.all((estimates >= 0) & (estimates <= 1))
        assert np.all((lower_bounds >= 0) & (lower_bounds <= 1))
        assert np.all((upper_bounds >= 0) & (upper_bounds <= 1))

        # Check confidence interval ordering
        valid_mask = ~np.isnan(estimates)
        if np.any(valid_mask):
            assert np.all(lower_bounds[valid_mask] <= estimates[valid_mask])
            assert np.all(estimates[valid_mask] <= upper_bounds[valid_mask])

    def test_tnr_sensitivity(self):
        """Test TNR sensitivity experiment."""
        tnr_values = [0.6, 0.8, 1.0]
        estimates, lower_bounds, upper_bounds = run_sensitivity_experiment(
            accuracy_values=tnr_values,
            fixed_accuracy=0.9,  # Fixed TPR
            vary_tpr=False,
            bootstrap_iterations=100,
            random_seed=42,
        )

        # Check output shapes
        assert len(estimates) == len(tnr_values)
        assert len(lower_bounds) == len(tnr_values)
        assert len(upper_bounds) == len(tnr_values)

    def test_poor_accuracy_handling(self):
        """Test handling of poor judge accuracy that causes estimation to fail."""
        # Use very poor accuracy values that should cause failures
        poor_values = [0.1, 0.2, 0.3]  # Very low accuracy
        estimates, lower_bounds, upper_bounds = run_sensitivity_experiment(
            accuracy_values=poor_values,
            fixed_accuracy=0.1,  # Also very low
            vary_tpr=True,
            bootstrap_iterations=50,
            random_seed=42,
        )

        # Should handle failures gracefully with NaN values
        assert len(estimates) == len(poor_values)
        # Some or all estimates might be NaN due to poor accuracy
        assert np.all(np.isnan(estimates) | ((estimates >= 0) & (estimates <= 1)))


class TestCreateExampleDataset:
    """Test cases for create_example_dataset function."""

    def test_all_scenarios(self):
        """Test all predefined scenarios."""
        scenarios = ["good_judge", "mediocre_judge", "biased_judge", "poor_judge"]

        for scenario in scenarios:
            test_labels, test_preds, unlabeled_preds = create_example_dataset(
                scenario=scenario, random_seed=42
            )

            # Check basic properties
            assert len(test_labels) == 100  # 50 positive + 50 negative
            assert len(test_preds) == 100
            assert len(unlabeled_preds) == 500

            # Check that all values are binary
            assert np.all(np.isin(test_labels, [0, 1]))
            assert np.all(np.isin(test_preds, [0, 1]))
            assert np.all(np.isin(unlabeled_preds, [0, 1]))

            # Check balanced test set
            assert np.sum(test_labels) == 50

    def test_reproducibility(self):
        """Test that same scenario and seed produce same results."""
        result1 = create_example_dataset(scenario="good_judge", random_seed=123)
        result2 = create_example_dataset(scenario="good_judge", random_seed=123)

        assert np.array_equal(result1[0], result2[0])
        assert np.array_equal(result1[1], result2[1])
        assert np.array_equal(result1[2], result2[2])

    def test_different_scenarios_differ(self):
        """Test that different scenarios produce different results."""
        good_result = create_example_dataset(scenario="good_judge", random_seed=42)
        poor_result = create_example_dataset(scenario="poor_judge", random_seed=42)

        # Results should be different (at least predictions should differ)
        assert not np.array_equal(good_result[1], poor_result[1])

    def test_invalid_scenario(self):
        """Test error handling for invalid scenario."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            create_example_dataset(scenario="invalid_scenario")

    def test_scenario_accuracy_properties(self):
        """Test that scenarios have expected accuracy properties."""
        # Good judge should have high accuracy
        test_labels, test_preds, _ = create_example_dataset(
            scenario="good_judge", random_seed=42
        )

        pos_mask = test_labels == 1
        neg_mask = test_labels == 0
        tpr = np.sum((test_labels == 1) & (test_preds == 1)) / np.sum(pos_mask)
        tnr = np.sum((test_labels == 0) & (test_preds == 0)) / np.sum(neg_mask)

        # Good judge should have reasonably high accuracy
        assert tpr > 0.8
        assert tnr > 0.8

        # Poor judge should have lower accuracy
        test_labels, test_preds, _ = create_example_dataset(
            scenario="poor_judge", random_seed=42
        )

        pos_mask = test_labels == 1
        neg_mask = test_labels == 0
        tpr_poor = np.sum((test_labels == 1) & (test_preds == 1)) / np.sum(pos_mask)
        tnr_poor = np.sum((test_labels == 0) & (test_preds == 0)) / np.sum(neg_mask)

        # Poor judge should have lower accuracy than good judge
        assert tpr_poor < tpr
        assert tnr_poor < tnr
