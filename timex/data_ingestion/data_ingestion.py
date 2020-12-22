import logging
from warnings import warn

import dateparser
import pandas as pd
from pandas import Series, DataFrame

from timex.data_preparation.data_preparation import add_diff_column

log = logging.getLogger(__name__)


def data_ingestion(param_config):
    """Retrieve the data at the URL in config_file_name and return it in a Pandas' DataFrame.
    Add diff columns and rename the columns if specified in the configuration.

    Parameters
    ----------
    param_config : dict
        A timex json configuration file.

    Returns
    -------
    df_ingestion : DataFrame
        Pandas dataframe storing the data loaded from the url in config_file_name, with optional diff columns and the
        correct names.
    """

    input_parameters = param_config["input_parameters"]

    # Extract parameters from parameters' dictionary.
    columns_to_load_from_url = input_parameters["columns_to_load_from_url"]
    source_data_url = input_parameters["source_data_url"]
    index_column_name = input_parameters["index_column_name"]
    try:
        freq = input_parameters["frequency"]
    except KeyError:
        freq = None

    log.info('Starting the data ingestion phase.')

    if columns_to_load_from_url == "":
        # Use all columns.
        df_ingestion = pd.read_csv(source_data_url)
    else:
        columns_to_read = list(columns_to_load_from_url.split(','))
        # We append [columns_to_read] to read_csv to maintain the same order of columns also in the df.
        df_ingestion = pd.read_csv(source_data_url, usecols=columns_to_read)[columns_to_read]

    # These are optional.
    if "datetime_column_name" in input_parameters:
        datetime_column_name = input_parameters["datetime_column_name"]
        log.debug(f"Parsing {datetime_column_name} as datetime column...")

        if "dateparser_options" in input_parameters:
            dateparser_options = input_parameters["dateparser_options"]
            df_ingestion[datetime_column_name] = df_ingestion[datetime_column_name].apply(
                lambda x: dateparser.parse(x, **dateparser_options)
            )
        else:
            df_ingestion[datetime_column_name] = df_ingestion[datetime_column_name].apply(
                lambda x: dateparser.parse(x)
            )

    df_ingestion.set_index(index_column_name, inplace=True, drop=True)

    log.debug(f"Removing duplicates rows from dataframe; keep the last...")
    df_ingestion = df_ingestion[~df_ingestion.index.duplicated(keep='last')]

    if "add_diff_column" in input_parameters:
        log.debug(f"Adding the diff columns...")
        targets = list(input_parameters["add_diff_column"].split(','))
        df_ingestion = add_diff_column(df_ingestion, targets)

    if "scenarios_names" in input_parameters:
        mappings = input_parameters["scenarios_names"]
        df_ingestion.reset_index(inplace=True)
        df_ingestion.rename(columns=mappings, inplace=True)
        try:
            df_ingestion.set_index(mappings.get(index_column_name), inplace=True)
        except KeyError:
            df_ingestion.set_index(index_column_name, inplace=True)

    log.info(f"Finished the data-ingestion phase. Some stats:\n"
             f"-> Number of rows: {len(df_ingestion)}\n"
             f"-> Number of columns: {len(df_ingestion.columns)}\n"
             f"-> Column names: {[*df_ingestion.columns]}\n"
             f"-> Number of missing data: {[*df_ingestion.isnull().sum()]}")

    df_ingestion = add_freq(df_ingestion, freq)
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
            return df
        else:
            if freq == "D":
                local_df.index = local_df.index.normalize()

            local_df = local_df.asfreq(freq=freq)
            return local_df
