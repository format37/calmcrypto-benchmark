"""Data quality checks for signal evaluation."""

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


# Signal-specific validation rules
SIGNAL_RULES = {
    'price': {'allow_zero': False, 'allow_negative': False},
    'rsi_raw': {'allow_zero': True, 'allow_negative': False, 'min': 0, 'max': 100},
    'open_interest': {'allow_zero': False, 'allow_negative': False},
    'total_borrow': {'allow_zero': True, 'allow_negative': False},
    'total_repay': {'allow_zero': True, 'allow_negative': False},
    'funding_rate': {'allow_zero': True, 'allow_negative': True},
}

# Signals to skip anomaly detection (derived signals have expected high variance)
SKIP_ANOMALY_CHECK = {'momentum', 'zscore', 'ratio'}


@dataclass
class QualityConfig:
    """Thresholds for data quality checks."""
    expected_interval_minutes: int = 5
    max_gap_minutes: int = 10            # Flag gaps > 2x expected
    max_nan_pct: float = 5.0             # Warn if > 5% NaN
    max_consecutive_dupes: int = 12      # 1 hour at 5-min intervals
    anomaly_zscore: float = 5.0          # Flag > 5 std from mean
    max_pct_change: float = 50.0         # Flag > 50% jumps


@dataclass
class QualityIssue:
    """Single data quality issue."""
    signal: str
    issue_type: str      # gap, nan, stale, anomaly, range
    severity: str        # warning, error
    timestamp: Optional[pd.Timestamp] = None
    message: str = ""
    value: Optional[float] = None


@dataclass
class SignalReport:
    """Quality report for one signal."""
    name: str
    total_rows: int
    nan_count: int = 0
    nan_pct: float = 0.0
    gap_count: int = 0
    stale_periods: int = 0
    anomaly_count: int = 0
    issues: List[QualityIssue] = field(default_factory=list)
    passed: bool = True


@dataclass
class QualityReport:
    """Overall quality report."""
    total_signals: int = 0
    passed: int = 0
    warnings: int = 0
    signals: Dict[str, SignalReport] = field(default_factory=dict)


class DataQualityChecker:
    """Run quality checks on signal data.

    Usage for debugging:
        checker = DataQualityChecker()
        report = checker.check_all(signals_dict, price_series)
    """

    def __init__(self, config: Optional[QualityConfig] = None):
        self.config = config or QualityConfig()

    # =========================================================================
    # BREAKPOINT: check_gaps - inspect 'gaps' df to see all timestamp issues
    # =========================================================================
    def check_gaps(self, series: pd.Series, name: str) -> List[QualityIssue]:
        """Check for unexpected time gaps between data points."""
        issues = []
        if not isinstance(series.index, pd.DatetimeIndex):
            return issues

        expected = timedelta(minutes=self.config.expected_interval_minutes)
        max_gap = timedelta(minutes=self.config.max_gap_minutes)

        time_diffs = series.index.to_series().diff()
        gaps = time_diffs[time_diffs > max_gap].dropna()

        # BREAKPOINT: inspect 'gaps' - shows all timestamps with gaps > max_gap
        for ts, diff in gaps.items():
            issues.append(QualityIssue(
                signal=name,
                issue_type='gap',
                severity='warning',
                timestamp=ts,
                message=f"{int(diff.total_seconds()/60)}min gap (expected {expected})",
                value=diff.total_seconds() / 60
            ))

        return issues

    # =========================================================================
    # BREAKPOINT: check_missing - inspect 'nan_count' and 'nan_pct'
    # =========================================================================
    def check_missing(self, series: pd.Series, name: str) -> Tuple[int, float, List[QualityIssue]]:
        """Check for NaN/missing values."""
        nan_count = series.isna().sum()
        nan_pct = (nan_count / len(series)) * 100 if len(series) > 0 else 0
        issues = []

        # BREAKPOINT: check nan_count and nan_pct values
        if nan_pct > self.config.max_nan_pct:
            issues.append(QualityIssue(
                signal=name,
                issue_type='nan',
                severity='warning',
                message=f"{nan_pct:.1f}% missing values ({nan_count} rows)",
                value=nan_pct
            ))

        return nan_count, nan_pct, issues

    # =========================================================================
    # BREAKPOINT: check_stale - inspect 'runs' to see consecutive duplicate stretches
    # =========================================================================
    def check_stale(self, series: pd.Series, name: str) -> Tuple[int, List[QualityIssue]]:
        """Check for stale data (too many consecutive identical values)."""
        issues = []
        stale_periods = 0
        clean = series.dropna()

        if len(clean) < 2:
            return 0, issues

        # Find runs of identical values
        is_same = clean == clean.shift(1)
        run_starts = (~is_same).cumsum()
        runs = clean.groupby(run_starts).size()
        long_runs = runs[runs > self.config.max_consecutive_dupes]

        # BREAKPOINT: inspect 'long_runs' - shows all stale data stretches
        for run_id, count in long_runs.items():
            stale_periods += 1
            # Find the start timestamp of this run
            run_mask = run_starts == run_id
            start_ts = clean[run_mask].index[0] if run_mask.any() else None

            issues.append(QualityIssue(
                signal=name,
                issue_type='stale',
                severity='warning',
                timestamp=start_ts,
                message=f"{count} consecutive identical values",
                value=float(count)
            ))

        return stale_periods, issues

    # =========================================================================
    # BREAKPOINT: check_anomalies - inspect 'outliers' and 'large_jumps'
    # =========================================================================
    def check_anomalies(self, series: pd.Series, name: str) -> List[QualityIssue]:
        """Check for statistical outliers and large jumps."""
        issues = []
        clean = series.dropna()

        if len(clean) < 10:
            return issues

        # Z-score outliers
        mean, std = clean.mean(), clean.std()
        if std > 0:
            zscores = (clean - mean) / std
            outliers = clean[zscores.abs() > self.config.anomaly_zscore]

            # BREAKPOINT: inspect 'outliers' - shows extreme values
            for ts, val in outliers.items():
                z = (val - mean) / std
                issues.append(QualityIssue(
                    signal=name,
                    issue_type='anomaly',
                    severity='warning',
                    timestamp=ts,
                    message=f"z-score {z:.1f} (value={val:.2f})",
                    value=val
                ))

        # Large percentage jumps
        pct_changes = clean.pct_change().abs() * 100
        large_jumps = pct_changes[pct_changes > self.config.max_pct_change]

        # BREAKPOINT: inspect 'large_jumps' - shows sudden value changes
        for ts, pct in large_jumps.items():
            issues.append(QualityIssue(
                signal=name,
                issue_type='anomaly',
                severity='warning',
                timestamp=ts,
                message=f"{pct:.1f}% jump",
                value=clean.loc[ts]
            ))

        return issues

    # =========================================================================
    # BREAKPOINT: check_range - inspect 'violations' for out-of-range values
    # =========================================================================
    def check_range(self, series: pd.Series, name: str) -> List[QualityIssue]:
        """Check signal-specific range rules (zeros, negatives, min/max)."""
        issues = []
        clean = series.dropna()

        # Get rules for this signal type (match by prefix)
        rules = None
        for pattern, r in SIGNAL_RULES.items():
            if name.startswith(pattern) or name == pattern:
                rules = r
                break

        if not rules:
            return issues  # No specific rules for this signal

        violations = []

        # Check zeros
        if not rules.get('allow_zero', True):
            zeros = clean[clean == 0]
            for ts, val in zeros.items():
                violations.append((ts, 'zero value not allowed', val))

        # Check negatives
        if not rules.get('allow_negative', True):
            negatives = clean[clean < 0]
            for ts, val in negatives.items():
                violations.append((ts, 'negative value not allowed', val))

        # Check min/max bounds
        if 'min' in rules:
            below_min = clean[clean < rules['min']]
            for ts, val in below_min.items():
                violations.append((ts, f'below min ({rules["min"]})', val))

        if 'max' in rules:
            above_max = clean[clean > rules['max']]
            for ts, val in above_max.items():
                violations.append((ts, f'above max ({rules["max"]})', val))

        # BREAKPOINT: inspect 'violations' - shows all range violations
        for ts, msg, val in violations:
            issues.append(QualityIssue(
                signal=name,
                issue_type='range',
                severity='warning',
                timestamp=ts,
                message=msg,
                value=val
            ))

        return issues

    def _should_skip_anomaly(self, name: str) -> bool:
        """Check if signal should skip anomaly detection."""
        return any(skip in name for skip in SKIP_ANOMALY_CHECK)

    # =========================================================================
    # BREAKPOINT: check_signal - step through each check type
    # =========================================================================
    def check_signal(self, signal: pd.Series, name: str) -> SignalReport:
        """Run all checks on a single signal."""
        report = SignalReport(name=name, total_rows=len(signal))

        # 1. Time gaps
        gap_issues = self.check_gaps(signal, name)
        report.gap_count = len(gap_issues)
        report.issues.extend(gap_issues)

        # 2. Missing values (skip warning for zscore signals - they have expected warmup NaN)
        nan_count, nan_pct, nan_issues = self.check_missing(signal, name)
        report.nan_count = nan_count
        report.nan_pct = nan_pct
        if 'zscore' not in name:  # zscore has expected NaN from rolling window
            report.issues.extend(nan_issues)

        # 3. Stale data
        stale_periods, stale_issues = self.check_stale(signal, name)
        report.stale_periods = stale_periods
        report.issues.extend(stale_issues)

        # 4. Anomalies (skip for derived signals with expected high variance)
        if not self._should_skip_anomaly(name):
            anomaly_issues = self.check_anomalies(signal, name)
            report.anomaly_count = len(anomaly_issues)
            report.issues.extend(anomaly_issues)

        # 5. Range checks
        range_issues = self.check_range(signal, name)
        report.issues.extend(range_issues)

        # BREAKPOINT: inspect 'report' - full report for this signal
        report.passed = len(report.issues) == 0
        return report

    # =========================================================================
    # BREAKPOINT: check_all - main entry point, inspect 'quality_report'
    # =========================================================================
    def check_all(
        self,
        signals: Dict[str, pd.Series],
        price: Optional[pd.Series] = None
    ) -> QualityReport:
        """Run quality checks on all signals.

        Args:
            signals: Dict of signal_name -> pandas Series
            price: Optional price series to check as well

        Returns:
            QualityReport with per-signal and overall summary
        """
        quality_report = QualityReport()

        # Add price to checks if provided
        all_series = dict(signals)
        if price is not None:
            all_series['price'] = price

        for name, series in all_series.items():
            signal_report = self.check_signal(series, name)
            quality_report.signals[name] = signal_report

            if signal_report.passed:
                quality_report.passed += 1
            else:
                quality_report.warnings += 1

        quality_report.total_signals = len(quality_report.signals)

        # BREAKPOINT: inspect 'quality_report' - full report with all signals
        return quality_report

    def print_summary(self, report: QualityReport) -> None:
        """Print quality summary to console."""
        print(f"\nData Quality: {report.total_signals} signals checked, "
              f"{report.passed} passed, {report.warnings} warnings")

        if report.warnings > 0:
            for name, sig_report in report.signals.items():
                if not sig_report.passed:
                    issue_types = {}
                    for issue in sig_report.issues:
                        issue_types[issue.issue_type] = issue_types.get(issue.issue_type, 0) + 1
                    summary = ", ".join(f"{v} {k}" for k, v in issue_types.items())
                    print(f"  {name}: {summary}")
