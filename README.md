# judgy

A Python library for estimating success rates when using LLM judges for evaluation.

[![PyPI version](https://badge.fury.io/py/judgy.svg)](https://badge.fury.io/py/judgy)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

When using Large Language Models (LLMs) as judges to evaluate other models or systems, the judge's own biases and errors can significantly impact the reliability of the evaluation. **judgy** provides tools to estimate the true success rate of your system by correcting for LLM judge bias, and bootstrapping to generate a confidence interval.

## Installation

### Basic Installation

```bash
pip install judgy
```

### With Plotting Support

```bash
pip install judgy[plotting]
```

### Development Installation

```bash
git clone https://github.com/ai-evals-course/judgy.git
cd judgy
pip install -e .[dev,plotting]
```

## Quick Start

```python
import numpy as np
from judgy import estimate_success_rate

# Your data: 1 = Pass, 0 = Fail
test_labels = [1, 1, 0, 0, 1, 0, 1, 0]      # Human labels on test set
test_preds = [1, 0, 0, 1, 1, 0, 1, 0]       # LLM judge predictions on test set  
unlabeled_preds = [1, 1, 0, 1, 0, 1, 0, 1]  # LLM judge predictions on unlabeled data

# Estimate true pass rate with 95% confidence interval
theta_hat, lower_bound, upper_bound = estimate_success_rate(
    test_labels=test_labels,
    test_preds=test_preds, 
    unlabeled_preds=unlabeled_preds
)

print(f"Estimated true pass rate: {theta_hat:.3f}")
print(f"95% Confidence interval: [{lower_bound:.3f}, {upper_bound:.3f}]")
```

## How It Works

The library implements a bias correction method based on the following steps:

1. **Judge Accuracy Estimation**: Calculate the LLM judge's True Positive Rate (TPR) and True Negative Rate (TNR) using labeled test data
2. **Correction**: Apply the correction formula to account for judge bias:
   ```
   θ̂ = (p_obs + TNR - 1) / (TPR + TNR - 1)
   ```
   where `p_obs` is the observed pass rate from the judge
3. **Bootstrap Confidence Intervals**: Use bootstrap resampling to quantify uncertainty in the estimate

## API Reference

### Core Function

#### `estimate_success_rate(test_labels, test_preds, unlabeled_preds, bootstrap_iterations=20000, confidence_level=0.95)`

Estimate true pass rate with bias correction and confidence intervals.

**Parameters:**
- `test_labels`: Array-like of 0/1 values (human labels on test set)
- `test_preds`: Array-like of 0/1 values (judge predictions on test set)  
- `unlabeled_preds`: Array-like of 0/1 values (judge predictions on unlabeled data)
- `bootstrap_iterations`: Number of bootstrap iterations (default: 20000)
- `confidence_level`: Confidence level between 0 and 1 (default: 0.95)

**Returns:**
- `theta_hat`: Point estimate of true pass rate
- `lower_bound`: Lower bound of confidence interval
- `upper_bound`: Upper bound of confidence interval

## Real-World Usage Pattern

```python
from judgy import estimate_success_rate

# Step 1: Collect human labels on a test set
test_labels = [...]  # Human evaluation: 1 = good, 0 = bad

# Step 2: Get LLM judge predictions on the same test set  
test_preds = [...]   # LLM judge predictions: 1 = good, 0 = bad

# Step 3: Get LLM judge predictions on your unlabeled data
unlabeled_preds = [...]  # LLM judge predictions on data you want to evaluate

# Step 4: Estimate the true pass rate
true_rate, lower, upper = estimate_success_rate(test_labels, test_preds, unlabeled_preds)

print(f"Your system's estimated true success rate: {true_rate:.1%}")
print(f"95% confidence interval: [{lower:.1%}, {upper:.1%}]")
```

## Requirements

- Python 3.8+
- numpy >= 1.20.0

## Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest tests/ --cov=judgy --cov-report=html
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The Rogan-Gladen correction method for bias correction in diagnostic tests
- Bootstrap methodology for confidence interval estimation
- The Python scientific computing ecosystem (NumPy, matplotlib)

## Support

If you encounter any issues or have questions, please:

1. Check the [documentation](README.md)
2. Search existing [issues](https://github.com/ai-evals-course/judgy/issues)
3. Create a new issue with a minimal reproducible example

---

**Note**: This library assumes that your LLM judge performs better than random chance (TPR + TNR > 1). If your judge's accuracy is too low, the correction method may not be applicable.
