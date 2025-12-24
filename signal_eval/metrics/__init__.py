"""Metric calculation modules for signal evaluation."""

from .information_coefficient import calculate_ic
from .lead_lag import lead_lag_analysis
from .hit_rate import calculate_hit_rate
from .granger import granger_test
from .rolling_power import rolling_signal_quality

__all__ = [
    'calculate_ic',
    'lead_lag_analysis',
    'calculate_hit_rate',
    'granger_test',
    'rolling_signal_quality'
]
