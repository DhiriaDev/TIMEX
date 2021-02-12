import logging

import dateparser
import pandas as pd
from pandas import DataFrame

log = logging.getLogger(__name__)


def ingest_timeseries(param_config: dict):
    """Retrieve the data at the URL specified in param_config (input parameters) and return it in a Pandas' DataFrame.
    This can be used for the initial data_ingestion, i.e. to ingest the initial time-series.

    Parameters
    ----------
    param_config : dict
        A dictionary corresponding to a TIMEX JSON configuration file.

    Returns
    -------
    df_ingestion : DataFrame
        Pandas DataFrame corresponding to the CSV files specified in param_config, with the various pre-processing steps
         applied. In particular, a frequency has been forced to the datetime index, NaN values have been interpolated.

    Notes
    -----
    In particular, the input_parameters sub-dictionary part of param_config will be used. In input_parameters, the
    following options has to be specified:

    - source_data_url: local or remote URL pointing to a CSV file;

    Additionally, some other parameters can be specified:

    - index_column_name: the name of the column to use as index for the DataFrame. If not specified the first one will
      be used. This column's values will be parsed with dateparser to obtain a DateTimeIndex;
    - frequency: if specified, the corresponding frequency will be imposed. Refer to
      https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases for a list of possible
      values. If not specified the frequency will be infered.
    - columns_to_load_from_url: comma-separated string of columns' names which will be read from the CSV file. If not
      specified, all columns will be read;
    - add_diff_column: comma-separated string of columns' names for which a corresponding column containing the diff
      values should be created. They will be created with the name `col_name'_diff. Note that the first row of the
      dataset will be discarded;
    - scenarios_names: dictionary of key-values (old_name: new_name) used to rename some columns in the CSV;
    - dateparser_options: dictionary of key-values which will be given to `dateparser.parse()`.

    Examples
    --------
    >>> timex_dict = {
    ...  "input_parameters": {
    ...  "source_data_url": "tests/test_datasets/covid_example_data_ingestion.csv",
    ...  "columns_to_load_from_url": "data,nuovi_positivi,terapia_intensiva",
    ...  "index_column_name": "data",
    ...  "add_diff_column": "terapia_intensiva",
    ...  "scenarios_names":
    ...    {
    ...      "data": "Date",
    ...      "nuovi_positivi": "Daily cases",
    ...      "terapia_intensiva": "Total intensive care",
    ...      "terapia_intensiva_diff": "Daily intensive care",
    ...    }
    ...  }
    ...}
    >>> ingest_timeseries(timex_dict)
                    Daily cases  Total intensive care  Daily intensive care
    Date
    2020-02-25           93                    35                   9.0
    2020-02-26           78                    36                   1.0
    ...                 ...                   ...                   ...
    2021-02-10        12956                  2128                 -15.0
    2021-02-11        15146                  2126                  -2.0

    [353 rows x 3 columns]
    """
    log.info('Starting the data ingestion phase.')
    input_parameters = param_config["input_parameters"]

    source_data_url = input_parameters['source_data_url']

    try:
        columns_to_load_from_url = input_parameters["columns_to_load_from_url"]
        columns_to_read = list(columns_to_load_from_url.split(','))
        # We append [columns_to_read] to read_csv to maintain the same order of columns also in the df.
        df_ingestion = pd.read_csv(source_data_url, usecols=columns_to_read)[columns_to_read]

    except (KeyError, ValueError):
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
        targets = list(input_parameters["add_diff_column"].split(','))
        log.debug(f"Adding the diff columns...")
        df_ingestion = add_diff_column(df_ingestion, targets)
    except KeyError:
        pass

    try:
        mappings = input_parameters["scenarios_names"]
        df_ingestion.reset_index(inplace=True)
        df_ingestion.rename(columns=mappings, inplace=True)
        try:
            df_ingestion.set_index(mappings.get(index_column_name), inplace=True)
        except KeyError:
            df_ingestion.set_index(index_column_name, inplace=True)
    except KeyError:
        pass

    try:
        freq = input_parameters["frequency"]
    except KeyError:
        freq = None

    df_ingestion = add_freq(df_ingestion, freq)
    df_ingestion = df_ingestion.interpolate()

    log.info(f"Finished the data-ingestion phase. Some stats:\n"
             f"-> Number of rows: {len(df_ingestion)}\n"
             f"-> Number of columns: {len(df_ingestion.columns)}\n"
             f"-> Column names: {[*df_ingestion.columns]}\n"
             f"-> Number of missing data: {[*df_ingestion.isnull().sum()]}")

    return df_ingestion


def ingest_additional_regressors(source_data_url, param_config):
    """Create a DataFrame from the specified source_data_url, to be used as additional regressors.

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
    """Add a frequency attribute to idx, through inference or directly.

    If `freq` is None, it is inferred.
    """
    local_df = df.copy()

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


def data_selection(data_frame, param_config):
    """This allows the user to select only a part of a time series in DataFrame, according to some criteria.

    The selected data is returned in a Pandas' DataFrame object.

    Parameters
    ----------
    data_frame : DataFrame
        Pandas dataframe storing the data loaded from the url in config_file_name
    param_config : dict
        A timex json configuration file.

    Returns
    -------
    df: DataFrame
        Pandas' DataFrame after the selection phase.
    """
    try:
        selection_parameters = param_config["selection_parameters"]
    except KeyError:
        log.debug(f"Selection phase not requested by user. Skip.")
        return data_frame

    log.info(f"Total amount of rows before the selection phase: {len(data_frame)}")

    if "column_name_selection" in selection_parameters and "value_selection" in selection_parameters:
        column_name = param_config['selection_parameters']['column_name_selection']
        value = param_config['selection_parameters']['value_selection']

        log.debug(f"Selection over column {column_name} with value = {value}")
        data_frame = data_frame.loc[data_frame[column_name] == value]

    if "init_datetime" in selection_parameters:
        # datetime_format = input_parameters["datetime_format"]

        init_datetime = dateparser.parse(selection_parameters['init_datetime'])

        log.debug(f"Selection over date, keep data after {init_datetime}")
        mask = (data_frame.index.to_series() >= init_datetime)
        data_frame = data_frame.loc[mask]

    if "end_datetime" in selection_parameters:
        # datetime_format = input_parameters["datetime_format"]

        end_datetime = dateparser.parse(selection_parameters['end_datetime'])

        log.debug(f"Selection over date, keep data before {end_datetime}")
        mask = (data_frame.index.to_series() <= end_datetime)
        data_frame = data_frame.loc[mask]

    log.info(f"Total amount of rows after the selection phase: {len(data_frame)}")
    return data_frame


def add_diff_column(data_frame: DataFrame, column_name_target_diff: [str], group_by: str = None):
    """Function for adding a 1-step diff column computed on the column_name_target_diff of the data frame.

    The function automatically removes the first row of the data_frame since the diff value is NaN.
    If the group_by parameter is specified, the data is grouped by that sub-index and then the diff
    is applied.

    Parameters
    ----------
    group_by
    data_frame : DataFrame
        Pandas dataframe to add the diff column on.
    column_name_target_diff : [str]
        Columns used to compute the 1-step diff
    group_by : str, optional
        If specified, data is grouped by this index and then the diff is applied.

    Returns
    -------
    df: Pandas dataframe with the new diff columns (name of the new columns is 'name_diff') and without the first row

    """
    log.info(f"Total number of rows before the add diff_columns phase: {len(data_frame)}")
    log.info(f"Total number of columns before the add diff_columns phase: {len(data_frame.columns)}")

    for target in column_name_target_diff:
        tmp = data_frame[target]
        name = target + "_diff"
        if group_by:
            data_frame[name] = tmp.groupby(group_by).diff()
        else:
            data_frame[name] = tmp.diff()

    if group_by:
        nan_rows = len(data_frame.index.get_level_values(group_by).unique())
    else:
        nan_rows = 1

    data_frame = data_frame.iloc[nan_rows:]

    log.info(f"Total number of rows after the add diff_columns phase: {len(data_frame)}")
    log.info(f"Total number of columns after the add diff_columns phase: {len(data_frame.columns)}")

    return data_frame