# CalmCrypto MCP Server

MCP (Model Context Protocol) server for CalmCrypto cryptocurrency signal evaluation and price prediction.

## Features

- **list_assets** - Get all available cryptocurrency assets
- **benchmark_all_assets** - Evaluate and rank all assets by signal predictability
- **signal_eval** - Deep evaluation of trading signals for a single asset
- **predict_price** - Predict price direction with probability
- **py_eval** - Execute Python code with pandas/numpy for data analysis
- **save_tool_notes** / **read_tool_notes** - Knowledge capture for tool usage
- **get_request_log** - Audit trail of tool calls

## Quick Start

### Local Development

```bash
# From the mcp/backend directory
cd backend

# Set up Python path to include parent project
export PYTHONPATH=/path/to/calmcrypto-benchmark

# Copy and edit environment file
cp ../.env.local.example .env.prod
# Edit .env with your credentials

# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
```

The server will be available at `http://localhost:8007/calmcrypto/`

### Docker Deployment

```bash
# From the mcp directory
cd /path/to/calmcrypto-benchmark/mcp

# Copy and configure environment
cp .env.local.example .env.prod
# Edit .env.prod with your credentials

# Deploy
./compose.prod.sh
```

## Tool Usage

### list_assets

Get all available cryptocurrency assets.

```python
list_assets(requester="my-agent")
```

Returns CSV with asset symbols (BTC, ETH, SOL, etc.)

### predict_price

Predict price direction with probability.

```python
predict_price(
    requester="my-agent",
    asset="BTC",
    top_n=5,      # Number of signals to use
    days=14       # Days of historical data
)
```

Returns predictions for 1h, 12h, and 24h timeframes with confidence levels.

### signal_eval

Deep evaluation of trading signals for a single asset.

```python
signal_eval(
    requester="my-agent",
    asset="ETH",
    days=7,
    top_n=10
)
```

Returns ranked signals with:
- Composite score (0-1)
- Information Coefficient
- Hit rate
- Granger causality significance
- Contrarian flag

### benchmark_all_assets

Evaluate all assets by signal predictability.

```python
benchmark_all_assets(
    requester="my-agent",
    days=7,
    top_n_assets=50  # 0 = all assets
)
```

Returns assets ranked by best composite score.

### py_eval

Execute Python code with data analysis libraries.

```python
py_eval(
    requester="my-agent",
    code="""
import pandas as pd
df = pd.read_csv(f'{CSV_PATH}/available_assets_abc123.csv')
print(df.head())
"""
)
```

Available in environment:
- `pd` - pandas
- `np` - numpy
- `CSV_PATH` - path to data folder

## Response Format

All data tools return a standardized format:

```
Data saved to CSV

File: filename.csv
Rows: 100
Size: 5.2 KB

Schema (JSON):
{
  "column1": "string",
  "column2": "float"
}

Sample (first row):
| column1 | column2 |
|---------|---------|
| value1  | 0.123   |

Python snippet to load:
```python
import pandas as pd
df = pd.read_csv('data/mcp-calmcrypto/filename.csv')
print(df.head())
```

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MCP_NAME` | Service name | calmcrypto |
| `PORT` | Server port | 8007 |
| `MCP_TOKENS` | Allowed auth tokens (comma-separated) | (none) |
| `MCP_REQUIRE_AUTH` | Require authentication | false |
| `GRAFANA_URL` | Grafana API URL | (required) |
| `GRAFANA_USER` | Grafana username | (required) |
| `GRAFANA_PASSWORD` | Grafana password | (required) |

## Data Storage

All outputs are saved to `data/mcp-calmcrypto/`:
- CSV files from list_assets, signal_eval, benchmark_all_assets
- JSON files from predict_price
- Request logs in `requests/` subdirectory
- Tool notes in `tool_notes/` subdirectory

## Network

The server joins the `mcp-shared` Docker network for integration with other MCP services.

## Health Check

```bash
curl http://localhost:8007/health
```

Returns: `{"status": "healthy", "service": "calmcrypto-mcp"}`
