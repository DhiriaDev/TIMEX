import numpy as np
import pandas as pd
from pandas import DataFrame
from statsmodels.tsa.stattools import grangercausalitytests


def calc_xcorr(target: str, ingested_data: DataFrame, max_lags: int, modes: [str] = ["pearson"]) -> dict:
    """
    Calculate the cross-correlation for the ingested data.
    Use the scenario column as target; the correlation is computed against all lags of all the other columns which
    include numbers. NaN values, introduced by the various shifts, are replaced with 0.

    Parameters
    ----------
    target : str
    Column which is used as target for the cross correlation.

    ingested_data : DataFrame
    Entire dataframe parsed from app

    max_lags : int
    Limit the analysis to max lags.

    modes : [str]
    Cross-correlation can be computed with different algorithms. The available choices are:
        `matlab_normalized`: same as using the MatLab function xcorr(x, y, 'normalized')
        `pearson` : use Pearson formula (NaN values are fillled to 0)
        `kendall`: use Kendall formula (NaN values are filled to 0)
        `spearman`: use Spearman formula (NaN values are filled to 0)

    Returns
    -------
    result : dict
    Dictionary with a Pandas DataFrame set for every indicated mode.
    Each DataFrame has the lags as index and the correlation value for each column.
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
                c = c[len(ingested_data) - 1 - max_lags:len(ingested_data) + max_lags]

                result[col] = c

            result.index -= max_lags

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
            for i in range(-max_lags, max_lags + 1):
                shifted = df_shifted(ingested_data, target, i)
                shifted.fillna(0, inplace=True)

                corr = [shifted[target].corr(other=shifted[col], method=mode) for col in columns]
                result.loc[i] = corr

        results[mode] = result

    return results


def calc_all_xcorr(ingested_data: DataFrame, param_config: dict) -> dict:
    """
    Compute, for every column in ingested_data (excluding the index) the cross-correlation of that series with respect
    to all others columns in ingested data.

    Parameters
    ----------
    ingested_data : DataFrame
        Pandas DataFrame for which the cross-correlation of all columns should be computed.

    max_lags : int
        Limit the cross-correlation to at maximum max_lags in the past and future (from -max_lags to max_lags)

    param_config : dict
        TIMEX configuration dictionary, needed to for xcorr_modes and xcorr_max_lags.
        xcorr_modes indicate the different algorithms which should be used to compute the xcorr.
        The available choices are:
            `matlab_normalized`: same as using the MatLab function xcorr(x, y, 'normalized')
            `pearson` : use Pearson formula (NaN values are fillled to 0)
            `kendall`: use Kendall formula (NaN values are filled to 0)
            `spearman`: use Spearman formula (NaN values are filled to 0)

        xcorr_max_lags is the number of lags, both in positive and negative direction, to which the cross-correlation
        calculations should be limited to.

    Returns
    -------
    dict
        Python dict with a key for every cross-correlation algorithm requested.
    """
    xcorr_max_lags = param_config['xcorr_parameters']['xcorr_max_lags']
    xcorr_modes = [*param_config['xcorr_parameters']["xcorr_mode"].split(",")]
    d = {}
    for col in ingested_data.columns:
        d[col] = calc_xcorr(col, ingested_data, max_lags=xcorr_max_lags, modes=xcorr_modes)

    return d