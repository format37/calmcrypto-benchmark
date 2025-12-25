# Data Quality Checks - Debugging Guide

Quick guide for stepping through data quality checks in Python debugger.

## Running Standalone

```python
from signal_eval.data_quality import DataQualityChecker, QualityConfig
from signal_eval.data_fetcher import DataFetcher
from signal_eval.signals import SignalRegistry

# Fetch data
fetcher = DataFetcher(demo=True, asset="SOL")
raw_data = fetcher.fetch_all(hours=168)

# Build signals
registry = SignalRegistry.from_raw_data(raw_data)
signals = registry.all_signals()
price = registry.get_price_series(raw_data)

# Run quality checks
checker = DataQualityChecker()
report = checker.check_all(signals, price)
checker.print_summary(report)
```

## Key Breakpoints

Each check method has a `# BREAKPOINT:` comment. Set breakpoints there to inspect:

### 1. Time Gaps (`check_gaps`)
```python
# BREAKPOINT: inspect 'gaps' df
gaps = time_diffs[time_diffs > max_gap].dropna()
```
**Inspect:** `gaps` - DataFrame of all timestamps where gap > 10 min

### 2. Missing Values (`check_missing`)
```python
# BREAKPOINT: check nan_count and nan_pct values
nan_count = series.isna().sum()
nan_pct = (nan_count / len(series)) * 100
```
**Inspect:** `nan_count`, `nan_pct`

### 3. Stale Data (`check_stale`)
```python
# BREAKPOINT: inspect 'long_runs'
long_runs = runs[runs > self.config.max_consecutive_dupes]
```
**Inspect:** `long_runs` - Series showing stretches of repeated values > 1 hour

### 4. Anomalies (`check_anomalies`)
```python
# BREAKPOINT: inspect 'outliers' and 'large_jumps'
outliers = clean[zscores.abs() > self.config.anomaly_zscore]
large_jumps = pct_changes[pct_changes > self.config.max_pct_change]
```
**Inspect:** `outliers` (z-score > 5), `large_jumps` (> 50% change)

### 5. Range Violations (`check_range`)
```python
# BREAKPOINT: inspect 'violations'
violations = [(ts, msg, val), ...]
```
**Inspect:** `violations` - list of (timestamp, message, value) tuples

### 6. Full Signal Report (`check_signal`)
```python
# BREAKPOINT: inspect 'report'
report.passed = len(report.issues) == 0
```
**Inspect:** `report` - SignalReport with all issues for one signal

### 7. Overall Report (`check_all`)
```python
# BREAKPOINT: inspect 'quality_report'
quality_report.total_signals = len(quality_report.signals)
```
**Inspect:** `quality_report` - full QualityReport with all signals

## Output Files

After running evaluation:
- `data_quality_summary.csv` - one row per signal, pass/fail status
- `data_quality_issues.csv` - detailed list of all issues found

## Config Thresholds

```python
QualityConfig(
    expected_interval_minutes=5,   # Expected time between rows
    max_gap_minutes=10,            # Flag gaps > this
    max_nan_pct=5.0,               # Warn if > 5% NaN
    max_consecutive_dupes=12,      # 1 hour of stale data
    anomaly_zscore=5.0,            # Flag extreme outliers
    max_pct_change=50.0            # Flag > 50% jumps
)
```

## Signals with Skipped Checks

**Anomaly detection skipped** for derived signals (expected high variance):
- `*_momentum` - percentage change signals
- `*_zscore` - z-score normalized signals
- `*_ratio` - ratio-based signals

**NaN warnings skipped** for:
- `*_zscore` - expected NaN from rolling window warmup period
