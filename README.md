# CalmCrypto Signal Evaluation System

A cryptocurrency signal evaluation tool that fetches trading data from Grafana/VictoriaMetrics, calculates predictive metrics, and generates interactive benchmark reports.

**[📈 Multi-Asset Report Demo](https://format37.github.io/calmcrypto-benchmark/assets/multi_asset_benchmark_report.html)**
**[📊 Single Asset Report Demo (OXT)](https://format37.github.io/calmcrypto-benchmark/assets/oxt_signal_benchmark_report.html)**

## MCP Server

An MCP (Model Context Protocol) server is available for programmatic access to signal evaluation and price prediction tools. See [mcp/README.md](mcp/README.md) for setup and usage.

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

### List Available Assets

```bash
python list_assets.py
```

Fetches all available assets from the Grafana API and saves them to `output/available_assets.csv`.

### Benchmark All Assets

```bash
# Benchmark all assets (may take time with 400+ assets)
python benchmark_all_assets.py

# Limit to first N assets
python benchmark_all_assets.py --top-n-assets 50

# Custom timeframe
python benchmark_all_assets.py --days 14
```

Ranks all assets by signal predictability and generates `output/asset_benchmark_summary.csv` and `output/multi_asset_benchmark_report.html`.

### Predict Price Direction

```bash
# Predict BTC price direction
python predict_price.py BTC

# Use more signals for prediction
python predict_price.py ETH --top-n 10

# Custom historical data period
python predict_price.py SOL --days 21
```

Predicts price direction (UP/DOWN) with probability for 1h, 12h, and 24h timeframes. Outputs to console and saves JSON to `output/prediction_{ASSET}.json`.

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

All signals are computed locally from 6 raw metrics fetched via API. Default asset: BTC.

### Raw Data Sources (6 API metrics)

| Source | PromQL Metric | Label | Description |
|--------|---------------|-------|-------------|
| price | `binance_price_usdt` | asset | Spot price in USDT |
| total_borrow | `binance_24h_total_borrow_usdt` | asset | 24h margin borrow volume |
| total_repay | `binance_24h_total_repay_usdt` | asset | 24h margin repay volume |
| rsi | `rsi{timeframe="3m", source="indicator_core"}` | symbol | RSI indicator (3min) |
| open_interest | `binance_futures_open_interest` | symbol | Futures open interest |
| funding_rate | `binance_futures_funding_rate` | symbol | Futures funding rate |

### Computed Signals (15 total)

| # | Signal | Description | Computation |
|---|--------|-------------|-------------|
| 1 | `borrow_repay_ratio` | Asset borrow / repay volume ratio | borrow / repay |
| 2 | `borrow_momentum` | 1-hour rate of change in borrow volume | borrow.pct_change(12) |
| 3 | `repay_momentum` | 1-hour rate of change in repay volume | repay.pct_change(12) |
| 4 | `rsi_raw` | Raw RSI indicator (3m timeframe) | direct |
| 5 | `rsi_zscore` | RSI standardized (z-score over 1-day window) | (rsi - rolling_mean) / rolling_std |
| 6 | `total_borrow` | Raw borrow volume for asset | direct |
| 7 | `total_repay` | Raw repay volume for asset | direct |
| 8 | `funding_rate` | Perpetual futures funding rate | direct |
| 9 | `open_interest` | Futures open interest | direct |
| 10 | `oi_momentum` | 1-hour rate of change in open interest | oi.pct_change(12) |
| 11 | `net_flow` | Borrow - Repay (net margin flow) | borrow - repay |
| 12 | `net_flow_momentum` | 1-hour rate of change in net flow | net_flow.pct_change(12) |
| 13 | `ratio_momentum` | 1-hour rate of change in borrow/repay ratio | ratio.pct_change(12) |
| 14 | `funding_zscore` | Funding rate standardized | (funding - mean) / std |
| 15 | `oi_zscore` | Open interest standardized | (oi - mean) / std |

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

## Price Prediction

The `predict_price.py` tool generates directional predictions for 1h, 12h, and 24h timeframes.

### Prediction Workflow

```
Raw Data (6 API metrics)
    ↓
Signal Computation (15 signals)
    ↓
Signal Evaluation (IC, hit rate, Granger, lead-lag)
    ↓
Top N Signal Selection (by composite score, Granger-filtered)
    ↓
Per-Signal Interpretation (config-driven rules)
    ↓
Timeframe Weighting (Gaussian by best_lag)
    ↓
Weighted Aggregation → Final Probability
```

### Signal Interpretation

Each signal type has a specific interpretation rule defined in `signal_eval/config.py`. The rules determine how to convert current signal values into bullish/bearish predictions.

| Type | Logic | Example |
|------|-------|---------|
| `threshold_contrarian` | Bullish below X, bearish above Y | RSI: <30 bullish, >70 bearish |
| `zscore_contrarian` | Extreme z-scores predict reversal | funding_zscore: \|z\|>2 = contrarian |
| `momentum_directional` | Trend-following over lookback window | borrow_momentum: positive = bullish |
| `level` | Simple threshold comparison | borrow_repay_ratio: >1 bullish |
| `level_with_extremes` | Directional normally, contrarian at extremes | funding_rate |

### Timeframe Matching

Signals are weighted by how well their optimal prediction horizon (`best_lag` from lead-lag analysis) matches each timeframe:

- Signal with `best_lag=12` (1 hour) → high weight for 1h prediction, low for 24h
- Gaussian weighting: `exp(-0.5 * ((best_lag - target) / sigma)^2)`
- Signals with <1% relevance are excluded from that timeframe

### Confidence Calculation

Confidence is derived from **signal agreement**, not probability:

| Factor | Description |
|--------|-------------|
| Agreement Ratio | % of weighted votes for majority direction (0.5-1.0) |
| Sample Size | Number of contributing signals (penalized if <5) |
| Confidence Score | Combined: `(agreement - 0.5) * 2 * 0.7 + sample_factor * 0.3` |

Labels: Very Low (<0.2), Low (<0.4), Medium (<0.6), High (<0.8), Very High (≥0.8)

### Customizing Signal Rules

Edit `config.json` or modify `signal_eval/config.py` to adjust interpretation rules:

```json
{
  "signal_rules": {
    "rsi_raw": {
      "type": "threshold_contrarian",
      "bullish_below": 30,
      "bearish_above": 70
    },
    "borrow_momentum": {
      "type": "momentum_directional",
      "lookback": 12,
      "invert": false
    }
  }
}
```

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
├── list_assets.py         # List available assets
├── benchmark_all_assets.py # Multi-asset benchmark
├── predict_price.py       # Price direction prediction
├── fetch.py               # Minimal API wrapper
├── calmcrypto_plot.py     # Original visualization script
├── signal_eval/           # Signal evaluation package
│   ├── config.py          # Configuration + signal rules
│   ├── data_fetcher.py    # Data fetching + demo mode
│   ├── signals.py         # Signal definitions
│   ├── evaluator.py       # Main evaluation engine
│   ├── interpreters.py    # Signal interpretation for prediction
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
