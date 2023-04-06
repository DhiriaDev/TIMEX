import pandas as pd
import statsmodels.api as sm
import numpy as np


def estimate_seasonality(series: pd.DataFrame):
    """
    Estimate seasonality in a time-series.
    Returns seasonality period. Returns 1 if no seasonality is found.
    """

    maxnlags = 30

    s = pd.Series(sm.tsa.pacf(series, method='ywm', nlags=maxnlags))
    s = np.abs(s)

    s[0] = 0
    s[1] = 0

    s = s[s > 2.58 / np.sqrt(len(series))].sort_values(ascending=False)

    if len(s) > 0:
        seasonality = s.index[0] - 1
    else:
        seasonality = 1

    return seasonality
