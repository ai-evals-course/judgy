"""
judgy: A Python library for estimating success rates when using LLM judges
for evaluation.

This package provides tools to estimate true pass rates from LLM judge predictions
by correcting bias and using bootstrap confidence intervals.
"""

from .core import estimate_success_rate

__version__ = "0.1.0"
__author__ = "judgy contributors"
__email__ = "your.email@example.com"

__all__ = ["estimate_success_rate"]
