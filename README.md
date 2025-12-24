# CalmCrypto Signal Evaluation System

A cryptocurrency signal evaluation tool that fetches trading data from Grafana/VictoriaMetrics, calculates predictive metrics, and generates interactive benchmark reports.

**[📊 Live Demo Report (SOL)](https://format37.github.io/calmcrypto-benchmark/assets/sol_signal_benchmark_report.html)**

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` with your Grafana credentials:
```
GRAFANA_URL=https://grafana.calmcrypto.app
GRAFANA_DS_UID=victoriametrics-uid
GRAFANA_USER=your_username
GRAFANA_PASSWORD=your_password
```

## Usage

### Basic Evaluation

```bash
# Run with live API data (7 days, top 10 signals, BTC default)
python -m signal_eval.run_evaluation

# Run with demo data (no API needed)
python -m signal_eval.run_evaluation --demo

# Analyze a specific asset
python -m signal_eval.run_evaluation --asset ETH
python -m signal_eval.run_evaluation --asset SOL

# Custom parameters
python -m signal_eval.run_evaluation --days 14 --top-n 15 --asset BTC --report
```

### Generate Interactive Report

```bash
# Fetch data and generate HTML report
python -m signal_eval.run_evaluation --days 7 --top-n 15 --report

# Generate report from existing output (no re-fetching)
python -m signal_eval.run_evaluation --from-output output/2025-12-24_123456
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--asset SYMBOL` | Asset to analyze (BTC, ETH, SOL, etc.). Default: BTC |
| `--demo` | Use synthetic demo data instead of live API |
| `--days N` | Number of days of historical data (default: 7) |
| `--top-n N` | Number of top signals to output (default: 10) |
| `--report` | Generate interactive HTML benchmark report |
| `--from-output DIR` | Load existing CSV data and generate report |
| `--include-rolling` | Save rolling signal quality CSVs |
| `--output-dir DIR` | Output directory (default: output/) |
| `--config FILE` | Config file path (default: config.json) |
| `--save-config` | Save default config to config.json |

## Output

Each run creates a timestamped folder in `output/` containing:

> 📖 **For programmatic usage**: See [CSV-README.md](CSV-README.md) for detailed CSV schema and trading agent integration.

- `summary.csv` - Ranked list of all signals with metrics
- `{signal}_data.csv` - Signal values with forward returns
- `{signal}_metrics.csv` - Detailed evaluation metrics
- `signal_benchmark_report.html` - Interactive report (with `--report`)

## Signals Evaluated

All signals are computed for the specified asset (default: BTC).

| Signal | Description |
|--------|-------------|
| `borrow_repay_ratio` | Asset borrow / repay volume ratio |
| `borrow_momentum` | 1-hour rate of change in borrow volume |
| `repay_momentum` | 1-hour rate of change in repay volume |
| `rsi_raw` | Raw RSI indicator (3m timeframe) |
| `rsi_zscore` | RSI standardized (z-score over 1-day window) |
| `total_borrow` | Raw borrow volume for asset |
| `total_repay` | Raw repay volume for asset |
| `funding_rate` | Perpetual futures funding rate |
| `funding_zscore` | Funding rate standardized |
| `open_interest` | Futures open interest |
| `oi_momentum` | 1-hour rate of change in open interest |
| `oi_zscore` | Open interest standardized |
| `net_flow` | Borrow - Repay (net margin flow) |
| `net_flow_momentum` | 1-hour rate of change in net flow |
| `ratio_momentum` | 1-hour rate of change in borrow/repay ratio |

## Evaluation Metrics

### Information Coefficient (IC)
Correlation between signal values and future price returns.

- **Pearson IC**: Linear correlation (-1 to 1)
- **Spearman IC**: Rank correlation, captures non-linear relationships
- **IC-IR (Information Ratio)**: IC mean / IC std - measures signal consistency

Higher absolute IC = stronger predictive relationship.

### Hit Rate
Percentage of times signal direction correctly predicts price direction.

- **Overall Hit Rate**: Raw directional accuracy
- **Hit Rate Bullish**: Accuracy when signal predicts up
- **Hit Rate Bearish**: Accuracy when signal predicts down
- **Effective Hit Rate**: `max(hit_rate, 1 - hit_rate)` - true predictive power
- **Is Contrarian**: True if hit rate < 50%

#### Understanding Hit Rate Values

| Hit Rate | Interpretation | Action |
|----------|----------------|--------|
| 50% | Random - no edge | Ignore signal |
| 55-65% | Direct signal | Trade with signal |
| 35-45% | **Contrarian signal** | Trade **opposite** to signal |
| >65% or <35% | Strong signal | High confidence trades |

#### Contrarian Signals

A hit rate **below 50%** means the signal reliably predicts the **opposite** direction:

```
Example: open_interest has 36% hit rate
├── When OI rises → price goes DOWN 64% of the time
├── When OI falls → price goes UP 64% of the time
└── Effective accuracy: 64% (by inverting)
```

**Key insight**: A 36% hit rate is just as valuable as 64% - you simply invert the signal. The only useless hit rate is exactly 50% (pure noise).

The system automatically:
1. Detects contrarian signals (`is_contrarian = True`)
2. Calculates `effective_hit_rate = max(hr, 1-hr)`
3. Uses effective hit rate in composite score

### Lead-Lag Analysis
Cross-correlation at different time lags to find which indicator leads price.

- **Best Lag**: Lag (in 5-min periods) with highest correlation
- **Lead-Lag Score**: Normalized score (0-1), higher = signal leads price

Positive lag = signal leads price (useful for prediction).

### Granger Causality
Statistical test for "does signal help predict price returns?"

- **P-Value**: Lower = more statistically significant
- **Significant**: True if p-value < 0.05
- **Granger Score**: Normalized score based on -log(p-value)

### Rolling Signal Quality
Tracks how signal predictive power changes over time.

- **Rolling IC**: IC calculated over sliding 1-day window
- **Rolling Hit Rate**: Hit rate over sliding window
- **Signal Score**: Composite of rolling metrics

Shows when signals are "hot" or "cold".

### Composite Score
Weighted combination of all metrics:

| Metric | Weight | Notes |
|--------|--------|-------|
| Spearman IC | 30% | Absolute value used |
| IC-IR | 25% | Information ratio |
| Effective Hit Rate | 20% | Handles contrarian signals |
| Lead-Lag Score | 15% | Rewards leading indicators |
| Granger Score | 10% | Statistical significance |

Note: Effective hit rate = `max(hit_rate, 1 - hit_rate)`, so both direct (>50%) and contrarian (<50%) signals are properly valued.

## Interactive Report

The HTML report includes:

1. **Signal Rankings** - Horizontal bar chart of composite scores
2. **Radar Chart** - Multi-metric comparison for top 5 signals
3. **IC vs Hit Rate** - Scatter plot showing signal quality distribution
4. **Metrics Heatmap** - Normalized metrics across all signals
5. **Lead-Lag Chart** - Which signals lead vs lag price
6. **Rolling Quality** - Time series of signal quality for top 5

## API Reference

```
# Grafana API endpoints
https://grafana.calmcrypto.app/api/search          # List dashboards
https://grafana.calmcrypto.app/api/datasources     # List data sources
```

## Project Structure

```
calmcrypto/
├── .env.example           # Environment template
├── dashboard.py           # Grafana API client
├── fetch.py               # Minimal API wrapper
├── calmcrypto_plot.py     # Original visualization script
├── signal_eval/           # Signal evaluation package
│   ├── config.py          # Configuration management
│   ├── data_fetcher.py    # Data fetching + demo mode
│   ├── signals.py         # Signal definitions
│   ├── evaluator.py       # Main evaluation engine
│   ├── output.py          # CSV output handling
│   ├── report.py          # HTML report generation
│   ├── loader.py          # Load from existing CSVs
│   ├── run_evaluation.py  # CLI entry point
│   └── metrics/           # Metric calculators
│       ├── information_coefficient.py
│       ├── lead_lag.py
│       ├── hit_rate.py
│       ├── granger.py
│       └── rolling_power.py
└── output/                # Generated results
```
