"""Signal Evaluation System for CalmCrypto.

This package provides tools for evaluating the predictive power of
trading signals against price movements.
"""

from .evaluator import SignalEvaluator
from .config import Config
from .signals import SignalRegistry

__all__ = ['SignalEvaluator', 'Config', 'SignalRegistry']
