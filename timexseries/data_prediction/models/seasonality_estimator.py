import pandas as pd
import statsmodels.api as sm


def estimate_seasonality(series: pd.DataFrame):
    """
    Estimate seasonality in a time-series.
    Returns seasonality period. Returns 1 if no seasonality is found.
    """
    try:
        s, confidence_intervals = sm.tsa.pacf(series.diff(1)[1:], method='ywm', nlags=int(len(series) / 2) - 5,
                                              alpha=0.01)
        s, confidence_intervals = pd.Series(abs(s)), pd.Series([abs(x[0]) for x in confidence_intervals])
        s[0] = 0
        s[1] = 0

        s = s[s > confidence_intervals].sort_values(ascending=False)
        if len(s) > 0:
            seasonality = s.index[0]
        else:
            seasonality = 1
    except:  # LinAlgError
        seasonality = 1

    return seasonality
