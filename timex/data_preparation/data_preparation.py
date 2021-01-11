import logging
from datetime import datetime

import dateparser
from pandas import DataFrame

log = logging.getLogger(__name__)


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
