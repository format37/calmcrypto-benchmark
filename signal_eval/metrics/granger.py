"""Granger Causality Test."""

import warnings
import pandas as pd
import numpy as np
from typing import Dict, Any

try:
    from statsmodels.tsa.stattools import grangercausalitytests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# Suppress statsmodels FutureWarning about verbose deprecation
warnings.filterwarnings('ignore', message='verbose is deprecated')


def granger_test(
    indicator: pd.Series,
    price: pd.Series,
    max_lag: int = 12
) -> Dict[str, Any]:
    """
    Test if indicator Granger-causes price movements.

    Lower p-value = stronger predictive relationship.

    Args:
        indicator: Signal/indicator series
        price: Price series
        max_lag: Maximum lag to test (default 12 periods = 1 hour at 5min)

    Returns:
        Dict with:
            - best_lag: Lag with lowest p-value
            - best_p_value: Lowest p-value found
            - significant: True if p-value < 0.05
            - granger_score: Score based on p-value (higher = more significant)
            - all_p_values: Dict of lag -> p-value
    """
    if not HAS_STATSMODELS:
        return {
            'best_lag': 0,
            'best_p_value': 1.0,
            'significant': False,
            'granger_score': 0.0,
            'all_p_values': {},
            'error': 'statsmodels not installed'
        }

    # Prepare data
    returns = price.pct_change()

    df = pd.concat([returns, indicator], axis=1).dropna()
    df.columns = ['price_return', 'indicator']

    if len(df) < max_lag * 3:
        return {
            'best_lag': 0,
            'best_p_value': 1.0,
            'significant': False,
            'granger_score': 0.0,
            'all_p_values': {}
        }

    try:
        # Run Granger causality test
        result = grangercausalitytests(
            df[['price_return', 'indicator']],
            maxlag=max_lag,
            verbose=False
        )

        # Extract p-values from F-test
        p_values = {}
        for lag in range(1, max_lag + 1):
            # result[lag][0] contains test statistics
            # 'ssr_ftest' is the F-test result: (F-stat, p-value, df_denom, df_num)
            p_values[lag] = result[lag][0]['ssr_ftest'][1]

        # Find best (lowest) p-value
        best_lag = min(p_values, key=p_values.get)
        best_p_value = p_values[best_lag]

        # Calculate score: -log10(p-value) normalized
        # p=0.05 -> score ~0.65, p=0.01 -> score ~1.0, p=0.5 -> score ~0.15
        if best_p_value > 0:
            granger_score = min(-np.log10(best_p_value) / 2, 1.0)
        else:
            granger_score = 1.0

        return {
            'best_lag': best_lag,
            'best_p_value': best_p_value,
            'significant': best_p_value < 0.05,
            'granger_score': granger_score,
            'all_p_values': p_values
        }

    except Exception as e:
        return {
            'best_lag': 0,
            'best_p_value': 1.0,
            'significant': False,
            'granger_score': 0.0,
            'all_p_values': {},
            'error': str(e)
        }
