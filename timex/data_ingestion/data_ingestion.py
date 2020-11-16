from warnings import warn

import pandas as pd
from pandas import Series


def data_ingestion(param_config):
    """Retrieve the data at the URL in config_file_name and return it in a Pandas' DataFrame.

    Parameters
    ----------
    param_config : dict
        A timex json configuration file.

    Returns
    -------
    df_ingestion : DataFrame
        Pandas dataframe storing the data loaded from the url in config_file_name.
    """

    verbose = param_config["verbose"]
    input_parameters = param_config["input_parameters"]

    # Extract parameters from parameters' dictionary.
    columns_to_load_from_url = input_parameters["columns_to_load_from_url"]
    source_data_url = input_parameters["source_data_url"]
    index_column_name = input_parameters["index_column_name"]
    try:
        freq = input_parameters["frequency"]
    except KeyError:
        freq = None

    if verbose == 'yes':
        print('------------------------------------------------------')
        print('Data_ingestion: starting the data ingestion phase.')

    columns_to_read = list(columns_to_load_from_url.split(','))
    # We append [columns_to_read] to read_csv to maintain the same order of columns also in the df.
    df_ingestion = pd.read_csv(source_data_url, usecols=columns_to_read)[columns_to_read]

    # These are optional.
    if "datetime_column_name" in input_parameters:
        datetime_column_name = input_parameters["datetime_column_name"]
        datetime_format = input_parameters["datetime_format"]
        # datetime_column = pd.to_datetime(df_ingestion[datetime_column_name], format=datetime_format)
        # datetime_column = add_freq(datetime_column)
        df_ingestion[datetime_column_name] = pd.to_datetime(df_ingestion[datetime_column_name], format=datetime_format)

    df_ingestion.set_index(index_column_name, inplace=True, drop=True)
    df_ingestion = add_freq(df_ingestion, freq)
    # df_ingestion.set_index(add_freq(df_ingestion.index, freq=freq))

    if verbose == 'yes':
        print('Data_ingestion: data frame (df) creation completed!')
        print('Data_ingestion: summary of statistics *** ')
        print('                |-> number of rows: ' + str(len(df_ingestion)))
        print('                |-> number of columns: ' + str(len(df_ingestion.columns)))
        print('                |-> column names: ' + str(list(df_ingestion.columns)))
        print('                |-> number of missing data: ' + str(list(df_ingestion.isnull().sum())))

    return df_ingestion


def add_freq(df, freq=None) -> Series:
    """Add a frequency attribute to idx, through inference or directly.

    If `freq` is None, it is inferred.
    """
    # TODO reorder this mess

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
            warn('no discernible frequency found for input data.')
            return df
        else:
            if freq == "D":
                local_df.index = local_df.index.normalize()

            local_df = local_df.asfreq(freq=freq)
            return local_df
