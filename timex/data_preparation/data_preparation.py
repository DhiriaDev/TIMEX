from datetime import datetime

from pandas import DataFrame


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
    verbose = param_config["verbose"]
    selection_parameters = param_config["selection_parameters"]
    input_parameters = param_config["input_parameters"]

    if verbose == 'yes':
        print("------------------------------------------------------")
        print('Data_selection: ' + 'starting the data selection phase')
        print('Data_selection: total amount of rows before the selection phase = ' + str(len(data_frame)))

    if "column_name_selection" in selection_parameters and "value_selection" in selection_parameters:
        column_name = param_config['selection_parameters']['column_name_selection']
        value = param_config['selection_parameters']['value_selection']

        print('Data_selection: ' + 'selection over column = ' + column_name + ' - with value = ' + str(value))
        data_frame = data_frame.loc[data_frame[column_name] == value]

    if "init_datetime" in selection_parameters:
        datetime_format = input_parameters["datetime_format"]

        init_datetime = datetime.strptime(selection_parameters['init_datetime'], datetime_format)

        print('Data_selection: ' + 'selection over date, data after ' + str(init_datetime))
        mask = (data_frame.index.to_series() >= init_datetime)
        data_frame = data_frame.loc[mask]

    if "end_datetime" in selection_parameters:
        datetime_format = input_parameters["datetime_format"]

        end_datetime = datetime.strptime(selection_parameters['end_datetime'], datetime_format)

        print('Data_selection: ' + 'selection over date, data before ' + str(end_datetime))
        mask = (data_frame.index.to_series() <= end_datetime)
        data_frame = data_frame.loc[mask]

    print('Data_selection: total amount of rows after the selection phase = ' + str(len(data_frame)))
    return data_frame


def add_diff_column(data_frame: DataFrame, column_name_target_diff: [str], group_by: str = None, verbose: str = 'yes'):
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
    verbose : str, optional
        Print details on the activities of the function (default is yes).

    Returns
    -------
    df: Pandas dataframe with the new diff columns (name of the new columns is 'name_diff') and without the first row

    """
    if verbose == 'yes':
        print('-------------------------------------------------')
        print('Adding_diff_column: ' + 'starting the diff  phase')
        print('Adding_diff_column: total number of rows before the adding phase = ' + str(len(data_frame)))
        print('                    total number of columns before the adding phase = ' + str(len(data_frame.columns)))

    for target in column_name_target_diff:
        tmp = data_frame[target]
        name = target + "_diff"
        if group_by:
            data_frame[name] = tmp.groupby(group_by).diff()
        else:
            data_frame[name] = tmp.diff()

    data_frame = data_frame.iloc[1:]

    if verbose == 'yes':
        print('Adding_diff_column: ' + 'completing the diff  phase')
        print('Adding_diff_column: total number of rows after the adding phase = ' + str(len(data_frame)))
        print('                    total number of columns after the adding phase = ' + str(len(data_frame.columns)))

    return data_frame
