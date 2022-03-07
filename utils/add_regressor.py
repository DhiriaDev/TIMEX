import logging

import dateparser
import pandas as pd
from pandas import DataFrame

log = logging.getLogger(__name__)


def ingest_additional_regressors(source_data_url, param_config):
    """Create a DataFrame from the data specified at source_data_url, to be used as additional regressors.

    Parameters
    ----------
    source_data_url : string
        URL of the CSV file to retrieve. Local or remote.
    param_config : dict
        A dictionary corresponding to a TIMEX JSON configuration file.

    Returns
    -------
    df_ingestion : DataFrame
        Pandas DataFrame corresponding to the CSV files specified in source_data_url.

    See Also
    --------
    ingest_timeseries :
        The logic is very similar, but this is used only to load a time-series which will be used as additional
        regressor for a multivariate prediction model.
    """

    input_parameters = param_config["input_parameters"]

    df_ingestion = pd.read_csv(source_data_url)

    try:
        index_column_name = input_parameters["index_column_name"]
    except KeyError:
        index_column_name = df_ingestion.columns[0]

    log.debug(f"Parsing {index_column_name} as datetime column...")

    if "dateparser_options" in input_parameters:
        dateparser_options = input_parameters["dateparser_options"]
        df_ingestion[index_column_name] = df_ingestion[index_column_name].apply(
            lambda x: dateparser.parse(x, **dateparser_options)
        )
    else:
        df_ingestion[index_column_name] = df_ingestion[index_column_name].apply(
            lambda x: dateparser.parse(x)
        )

    df_ingestion.set_index(index_column_name, inplace=True, drop=True)

    log.debug(f"Removing duplicates rows from dataframe; keep the last...")
    df_ingestion = df_ingestion[~df_ingestion.index.duplicated(keep='last')]

    try:
        freq = input_parameters["frequency"]
    except KeyError:
        freq = None

    df_ingestion = add_freq(df_ingestion, freq)
    df_ingestion = df_ingestion.interpolate()

    return df_ingestion


def add_freq(df, freq=None) -> DataFrame:
    """Add a frequency to the index of df. Pandas DatetimeIndex have a `frequency` attribute; this function tries to
    assign a value to that attribute.

    If the index of df is a DatetimeIndex, then this function is guaranteed to return a DataFrame with the `frequency`
    attribute set. If it is not possible to assign the frequency, datetime may be normalized (i.e. keep only the date
    part and remove hour) in order to obtain days.

    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame on which a frequency will be added.

    freq : str, optional
        If this attribute is specified, then the corresponding frequency will be forced on the DataFrame. If it is not
        specified, than, the frequency will be estimated.
        `freq` should be a so called 'offset alias'; the possible values can be found at
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases

    Returns
    -------
    local_df : DataFrame
        df with the DatetimeIndex.freq set; if df did not have a DatetimeIndex, then df is returned unmodified.

    Examples
    --------
    >>> dates = [pd.Timestamp(datetime(year=2020, month=1, day=1, hour=10, minute=00)),
    ...          pd.Timestamp(datetime(year=2020, month=1, day=2, hour=12, minute=21)),
    ...          pd.Timestamp(datetime(year=2020, month=1, day=3, hour=13, minute=30)),
    ...          pd.Timestamp(datetime(year=2020, month=1, day=4, hour=11, minute=32))]
    >>> df = pd.DataFrame(data={"a": [0, 1, 2, 3]}, index=dates)
    >>> df
                         a
    2020-01-01 10:00:00  0
    2020-01-02 12:21:00  1
    2020-01-03 13:30:00  2
    2020-01-04 11:32:00  3

    `df` does not have a fixed frequency in its DatetimeIndex:

    >>> df.index.freq
    None

    Try to apply it:

    >>> df_with_freq = add_freq(df)
    >>> df_with_freq
                a
    2020-01-01  0
    2020-01-02  1
    2020-01-03  2
    2020-01-04  3

    Hours have been removed. Check the frequency of the index:

    >>> df_with_freq.index.freq
    <Day>
    """
    local_df = df.copy()

    # Check if df has a DatetimeIndex. If not, return without doing anything.
    try:
        i = local_df.index.freq
    except:
        return local_df

    # Df has already a freq. Don't do anything.
    if local_df.index.freq is not None:
        return local_df

    if freq is not None:
        if freq == 'D':
            local_df.index = local_df.index.normalize()

        local_df = local_df.asfreq(freq=freq)
        return local_df

    if freq is None:
        freq = pd.infer_freq(local_df.index)

        if freq is None:
            local_df.index = local_df.index.normalize()
            freq = pd.infer_freq(local_df.index)

        if freq is None:
            log.warning(f"No discernible frequency found for the dataframe.")
            freq = "D"

        local_df = local_df.asfreq(freq=freq)
        return local_df

