"""Signal research report generation module.

Provides interactive Plotly charts for visualizing how trading signals
correlate with price movements over configurable time periods.

Usage:
    python -m signal_research.research_report --assets BTC ETH --days 90 --step 5m

Modules:
    signal_events: Generate discrete UP/DOWN events from continuous signals
    chart_builder: Build multi-subplot Plotly charts with signal markers
    exporters: Export raw data to CSV and consolidated XLS
    research_report: CLI entry point and orchestration
"""

from .signal_events import generate_signal_events, generate_all_signal_events
from .chart_builder import ResearchChartBuilder
from .exporters import export_raw_data_csv, export_signal_events_csv, export_consolidated_xls

__all__ = [
    'generate_signal_events',
    'generate_all_signal_events',
    'ResearchChartBuilder',
    'export_raw_data_csv',
    'export_signal_events_csv',
    'export_consolidated_xls',
]
