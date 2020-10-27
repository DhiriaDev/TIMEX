from datetime import datetime


def data_selection(data_frame, param_config):
    """This allows the user to select only a part of a time series in DataFrame, according to some criteria.

    The selected data is returned in a Pandas' DataFrame object.

    Parameters
    ----------
    data_frame : DataFrame
        Pandas dataframe storing the data loaded from the url in config_file_name
    param_config : dict, optional
        A timex json configuration file.

    Returns
    -------
    df: DataFrame
        Pandas' DataFrame after the selection phase.
    """
    verbose = param_config["verbose"]
    if verbose == 'yes':
        print('data_selection: ' + 'starting the data selection phase')
        print('data_selection: total amount of rows before the selection phase = ' + str(len(data_frame)))

    column_name = param_config['selection_parameters']['column_name_selection']
    value = param_config['selection_parameters']['value_selection']
    column_name_datetime = param_config['input_parameters']['datetime_column_name']

    datetime_format = param_config["input_parameters"]["datetime_format"]
    init_datetime = datetime.strptime(param_config['selection_parameters']['init_datetime'],
                                      datetime_format)
    end_datetime = datetime.strptime(param_config['selection_parameters']['end_datetime'],
                                         datetime_format)

    if column_name and value:
        print('data_selection: ' + 'selection over column = ' + column_name + ' - with value = ' + value)
        data_frame = data_frame.loc[data_frame[column_name] == value]

    if init_datetime and column_name_datetime:
        mask = (data_frame[column_name_datetime] >= init_datetime)
        data_frame = data_frame.loc[mask]

    if end_datetime and column_name_datetime:
        mask = (data_frame[column_name_datetime] <= end_datetime)
        data_frame = data_frame.loc[mask]

    print('data_selection: total amount of rows after the selection phase = ' + str(len(data_frame)))
    print('data_selection:    from ' + str(init_datetime) + ' to ' + str(end_datetime))
    return data_frame


def add_diff_column(data_frame, column_name_target_diff, name_diff_column=None, verbose='yes', ):
    """Function for adding a 1-step diff column computed on the column_name_target_diff of the data frame.

    The function automatically removes the first row of the data_frame since the diff value is nan
    If the name_diff_column parameter is specified, it is used as name of the new column.
    Otherwise, the name 'diff' is used

    Parameters
    ----------
    data_frame : DataFrame
       Pandas dataframe storing the data loaded from the url in config_file_name
    column_name_target_diff : str
        ???
    name_diff_column : str, optional
        The column where the selection with 'value' is applied
    verbose : str, optional
        Print details on the activities of the function (default is yes).

    Returns
    -------
    df: Pandas dataframe with the new diff column and without the first row

    """
    if verbose == 'yes':
        print('adding_diff_column: ' + 'starting the diff  phase')
        print('adding_diff_column: total number of rows before the adding phase = ' + str(len(data_frame)))
        print('                    total number of columns before the adding phase = ' + str(len(data_frame.columns)))

    tmp = data_frame[column_name_target_diff]
    if name_diff_column:
        data_frame[name_diff_column] = tmp.diff()
    else:
        data_frame[column_name_target_diff] = tmp.diff()

    data_frame = data_frame.iloc[1:]

    if verbose == 'yes':
        print('adding_diff_column: ' + 'completing the diff  phase')
        print('adding_diff_column: total number of rows after the adding phase = ' + str(len(data_frame)))
        print('                    total number of columns after the adding phase = ' + str(len(data_frame.columns)))

    return data_frame
