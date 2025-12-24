"""Configuration management for signal evaluation."""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict


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
