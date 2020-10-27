import pandas as pd


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

    if verbose == 'yes':
        print('data_ingestion: ' + 'starting the data ingestion phase')

    columns_to_read = list(columns_to_load_from_url.split(','))
    df_ingestion = pd.read_csv(source_data_url, usecols=columns_to_read)

    # These are optional.
    if "datetime_column_name" in input_parameters:
        datetime_column_name = input_parameters["datetime_column_name"]
        datetime_format = input_parameters["datetime_format"]
        df_ingestion[datetime_column_name] = pd.to_datetime(df_ingestion[datetime_column_name], format=datetime_format)

    df_ingestion.set_index(index_column_name, inplace=True, drop=True)

    if verbose == 'yes':
        print('data_ingestion: ' + 'data frame (df) creation completed!')
        print('data_ingestion: summary of statistics *** ')
        print('                |-> number of rows: ' + str(len(df_ingestion)))
        print('                |-> number of columns: ' + str(len(df_ingestion.columns)))
        print('                |-> column names: ' + str(list(df_ingestion.columns)))
        print('                |-> number of missing data: ' + str(list(df_ingestion.isnull().sum())))

    return df_ingestion

