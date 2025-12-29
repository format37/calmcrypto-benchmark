"""Configuration management for signal evaluation."""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict


def _default_signal_rules() -> Dict[str, dict]:
    """Default interpretation rules for each signal type."""
    return {
        # Threshold-based contrarian (RSI-style)
        'rsi_raw': {
            'type': 'threshold_contrarian',
            'bullish_below': 30,
            'bearish_above': 70,
            'neutral_range': [35, 65]
        },
        # Z-score contrarian (extreme values predict reversal)
        'rsi_zscore': {
            'type': 'zscore_contrarian',
            'threshold': 2.0
        },
        'funding_zscore': {
            'type': 'zscore_contrarian',
            'threshold': 2.0
        },
        'oi_zscore': {
            'type': 'zscore_contrarian',
            'threshold': 2.0
        },
        # Momentum directional (trend-following)
        'borrow_momentum': {
            'type': 'momentum_directional',
            'lookback': 12
        },
        'repay_momentum': {
            'type': 'momentum_directional',
            'lookback': 12,
            'invert': True  # Less repay = bullish
        },
        'net_flow_momentum': {
            'type': 'momentum_directional',
            'lookback': 12
        },
        'ratio_momentum': {
            'type': 'momentum_directional',
            'lookback': 12
        },
        'oi_momentum': {
            'type': 'momentum_directional',
            'lookback': 12
        },
        # Level-based
        'borrow_repay_ratio': {
            'type': 'level',
            'bullish_above': 1.0,
            'bearish_below': 0.9
        },
        'net_flow': {
            'type': 'level',
            'bullish_above': 0,
            'bearish_below': 0
        },
        # Level with extremes (normal directional, contrarian at extremes)
        'funding_rate': {
            'type': 'level_with_extremes',
            'bullish_above': 0,
            'bearish_below': 0,
            'extreme_threshold': 0.0005,  # 0.05% funding = extreme
            'extreme_contrarian': True
        },
        # Raw values - use momentum detection
        'total_borrow': {
            'type': 'momentum_directional',
            'lookback': 12
        },
        'total_repay': {
            'type': 'momentum_directional',
            'lookback': 12,
            'invert': True
        },
        'open_interest': {
            'type': 'momentum_directional',
            'lookback': 12
        }
    }


@dataclass
class Config:
    """Configuration for signal evaluation system."""

    # Number of top signals to select
    top_n: int = 10

    # Asset to analyze
    asset: str = "BTC"

    # Data fetching settings
    data_hours: int = 168  # 7 days
    step: str = "5m"

    # Forward periods to test (in 5-min intervals)
    # 1 = 5min, 12 = 1hr, 48 = 4hr, 288 = 1day
    forward_periods: List[int] = field(default_factory=lambda: [1, 12, 48, 288])

    # Composite score weights (from benchmark.md)
    weights: Dict[str, float] = field(default_factory=lambda: {
        'ic': 0.30,           # Information Coefficient (Spearman)
        'ic_ir': 0.25,        # IC Information Ratio
        'hit_rate': 0.20,     # Directional accuracy
        'lead_lag': 0.15,     # Lead-lag correlation score
        'granger': 0.10       # Granger causality score
    })

    # Metric calculation settings
    rolling_window: int = 288  # 1 day at 5min intervals
    max_lag: int = 48          # Max lag for lead-lag analysis (4 hours)
    granger_max_lag: int = 12  # Max lag for Granger test

    # Thresholds
    granger_significance: float = 0.05  # p-value threshold
    composite_threshold: float = 0.0    # Minimum composite score to include

    # Output settings
    output_dir: str = "output"

    # Per-signal interpretation rules for prediction
    signal_rules: Dict[str, dict] = field(default_factory=_default_signal_rules)

    @classmethod
    def load(cls, path: str = "config.json") -> 'Config':
        """Load config from JSON file."""
        config_path = Path(path)
        if config_path.exists():
            with open(config_path, 'r') as f:
                data = json.load(f)
            return cls(**data)
        return cls()

    def save(self, path: str = "config.json") -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    def forward_period_labels(self) -> Dict[int, str]:
        """Get human-readable labels for forward periods."""
        labels = {}
        for p in self.forward_periods:
            minutes = p * 5
            if minutes < 60:
                labels[p] = f"{minutes}min"
            elif minutes < 1440:
                labels[p] = f"{minutes // 60}hr"
            else:
                labels[p] = f"{minutes // 1440}day"
        return labels
