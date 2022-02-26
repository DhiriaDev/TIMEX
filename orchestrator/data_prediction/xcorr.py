import numpy as np
import pandas as pd
from pandas import DataFrame
from statsmodels.tsa.stattools import grangercausalitytests


def calc_xcorr(target: str, ingested_data: DataFrame, max_lags: int, modes: [str] = ["pearson"]) -> dict:
    """
    Calculate the cross-correlation for the `ingested data`.
    Use `target` time-series column as target; the correlation is computed against all lags of all the other columns
    which include numbers. NaN values, introduced by the various shifts, are replaced with 0.

    Parameters
    ----------
    target : str
        Column which is used as target for the cross correlation.

    ingested_data : DataFrame
        DataFrame which contains the various time-series, one for column.

    max_lags : int
        Limit the analysis to max lags.

    modes : [str]
        Cross-correlation can be computed with different algorithms. The available choices are:

        - `matlab_normalized`: same as using the MatLab function xcorr(x, y, 'normalized')
        - `pearson` : use Pearson formula (NaN values are fillled to 0)
        - `kendall`: use Kendall formula (NaN values are filled to 0)
        - `spearman`: use Spearman formula (NaN values are filled to 0)

    Returns
    -------
    result : dict
        Dictionary with a Pandas DataFrame set for every indicated mode.
        Each DataFrame has the lags as index and the correlation value for each column.

    Examples
    --------
    Create some sample time-series.
    >>> dates = pd.date_range('2000-01-01', periods=30)  # Last index is 2000-01-30
    >>> ds = pd.DatetimeIndex(dates, freq="D")
    >>>
    >>> x = np.linspace(0, 2 * np.pi, 60)
    >>> y = np.sin(x)
    >>>
    >>> np.random.seed(0)
    >>> noise = np.random.normal(0, 2.0, 60)
    >>> y = y + noise
    >>>
    >>> a = y[:30]
    >>> b = y[5:35]
    >>>
    >>> timeseries_dataframe = DataFrame(data={"a": a, "b": b}, index=ds)

    Compute the cross-correlation:
    >>> calc_xcorr("a", timeseries_dataframe, 7, ["pearson"])
    {'pearson':  b
             -7  0.316213
             -6 -0.022288
             -5  0.112483
             -4 -0.268724
             -3  0.105511
             -2  0.178658
             -1  0.101505
              0  0.051641
              1 -0.360475
              2 -0.074952
              3 -0.047689
              4 -0.252324
              5  0.796120
              6 -0.170558
              7 -0.009305
    }

    This is expected; the biggest value of cross-correlation is at index `5`. It is true that `b` is exactly time-series
    `a`, but shifted forward of `5` lags.
    """

    def df_shifted(df, _target=None, lag=0):
        if not lag and not _target:
            return df
        new = {}
        for c in df.columns:
            if c == _target:
                new[c] = df[_target]
            else:
                new[c] = df[c].shift(periods=lag)
        return pd.DataFrame(data=new)

    columns = ingested_data.columns.tolist()
    columns = [elem for elem in columns if ingested_data[elem].dtype != str and elem != target]

    results = {}
    for mode in modes:
        result = DataFrame(columns=columns, dtype=np.float64)
        if mode == 'matlab_normalized':
            lags_to_displace = max_lags if max_lags < len(ingested_data) else len(ingested_data) - 1
            for col in columns:
                x = ingested_data[target]
                y = ingested_data[col]

                c = np.correlate(x, y, mode="full")

                # This is needed to obtain the same result of the MatLab `xcorr` function with normalized results.
                # You can find the formula in the function pyplot.xcorr; however, here the property
                # sqrt(x*y) = sqrt(x) * sqrt(y)
                # is applied in order to avoid overflows if the ingested values are particularly high.
                den = np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y))
                c = np.divide(c, den)

                # This assigns the correct indexes to the results.
                if len(ingested_data) > max_lags:
                    c = c[len(ingested_data) - 1 - max_lags:len(ingested_data) + max_lags]

                result[col] = c

            result.index -= lags_to_displace

        elif mode == 'granger':
            for col in columns:
                granger_max_lags = int(len(ingested_data) / 3) - 1
                granger_max_lags = granger_max_lags if granger_max_lags < max_lags else max_lags

                # Trick to compute both negative and positive lags
                df = ingested_data[[col, target]]
                granger_result = grangercausalitytests(df, maxlag=granger_max_lags, verbose=False)
                for i in granger_result:
                    result.loc[-i, col] = 1 - granger_result[i][0]['params_ftest'][1]

                df = ingested_data[[target, col]]
                granger_result = grangercausalitytests(df, maxlag=granger_max_lags, verbose=False)
                for i in granger_result:
                    result.loc[i, col] = 1 - granger_result[i][0]['params_ftest'][1]

            result.sort_index(inplace=True)

        else:
            lags_to_displace = max_lags if max_lags < len(ingested_data) else len(ingested_data)
            for i in range(-lags_to_displace, lags_to_displace + 1):
                shifted = df_shifted(ingested_data, target, i)
                shifted.fillna(0, inplace=True)

                corr = [shifted[target].corr(other=shifted[col], method=mode) for col in columns]
                result.loc[i] = corr

        results[mode] = result

    return results


def calc_all_xcorr(ingested_data: DataFrame, param_config: dict) -> dict:
    """
    Compute, for every column in `ingested_data` (excluding the index) the cross-correlation of that series with respect
    to all others columns in ingested data.

    Parameters
    ----------
    ingested_data : DataFrame
        Pandas DataFrame for which the cross-correlation of all columns should be computed.

    param_config : dict
        TIMEX configuration dictionary, needed to for `xcorr_parameters`.
        In the `xcorr_parameters` sub-dictionary, `xcorr_modes` and `xcorr_max_lags` will be used.
        `xcorr_modes` indicate the different algorithms which should be used to compute the cross-correlation.
        The available choices are:

        - `matlab_normalized`: same as using the MatLab function xcorr(x, y, 'normalized')
        - `pearson` : use Pearson formula (NaN values are fillled to 0)
        - `kendall`: use Kendall formula (NaN values are filled to 0)
        - `spearman`: use Spearman formula (NaN values are filled to 0)

        `xcorr_max_lags` is the number of lags, both in positive and negative direction, to which the cross-correlation
        calculations should be limited to.

    Returns
    -------
    dict
        Python dict with a key for every time-series in `ingested_data`; every key will correspond to another dictionary
        with one entry for each cross-correlation algorithm requested.

    Examples
    --------
    Create sample data.
    >>> dates = pd.date_range('2000-01-01', periods=30)  # Last index is 2000-01-30
    >>> ds = pd.DatetimeIndex(dates, freq="D")
    >>>
    >>> x = np.linspace(0, 2 * np.pi, 60)
    >>> y = np.sin(x)
    >>> np.random.seed(0)
    >>> noise = np.random.normal(0, 2.0, 60)
    >>> y = y + noise
    >>>
    >>> a = y[:30]
    >>> b = y[2:32]
    >>> c = y[4:34]
    >>>
    >>> timeseries_dataframe = DataFrame(data={"a": a, "b": b, "c": c}, index=ds)

    Compute the cross-correlation on this DataFrame:
    >>> param_config = {
    >>>     "xcorr_parameters": {
    >>>         "xcorr_max_lags": 2,
    >>>         "xcorr_mode": "pearson,matlab_normalized"
    >>>     }
    >>> }
    >>> calc_all_xcorr(timeseries_dataframe, param_config)
    {'a': {'pearson':            b         c
                             -2 -0.252086  0.117286
                             -1  0.006370  0.064624
                              0 -0.011866 -0.290049
                              1 -0.115114 -0.091762
                              2  0.951782 -0.024158,
           'matlab_normalized':  b         c
                             -2  0.109634  0.287681
                             -1  0.314318  0.239430
                             0   0.319016  0.008663
                             1   0.244525  0.197663
                             2   0.965095  0.260254},

     'b': {'pearson':            a         c
                             -2  0.998491 -0.353341
                             -1 -0.085531 -0.007476
                              0 -0.011866  0.048841
                              1  0.013242 -0.092448
                              2 -0.258411  0.895226,
           'matlab_normalized':  a         c
                             -2  0.965095 -0.063331
                             -1  0.244525  0.177921
                             0   0.319016  0.252201
                             1   0.314318  0.183260
                             2   0.109634  0.862899},

     'c': {'pearson':            a         b
                             -2  0.076014  0.929572
                             -1 -0.013978 -0.026488
                              0 -0.290049  0.048841
                              1  0.038452 -0.043913
                              2  0.125275 -0.354749,
           'matlab_normalized':  a         b
                             -2  0.260254  0.862899
                             -1  0.197663  0.183260
                             0   0.008663  0.252201
                             1   0.239430  0.177921
                             2   0.287681 -0.063331}}
    """
    xcorr_max_lags = param_config['xcorr_parameters']['xcorr_max_lags']
    xcorr_modes = [*param_config['xcorr_parameters']["xcorr_mode"].split(",")]
    d = {}
    for col in ingested_data.columns:
        d[col] = calc_xcorr(col, ingested_data, max_lags=xcorr_max_lags, modes=xcorr_modes)

    return d