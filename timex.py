# LIST OF REQUIRED PACKAGES (to be imported):
#       - pandas
#       - numpy
#       - plotly
#       - psutil
#       - kaleido

def data_ingestion(config_file_name, verbose='yes'):
    """Retrieve the data at the URL in config_file_name and return it in a Pandas' DataFrame.

    Parameters
    ----------
    config_file_name : str
        Filename of the json config file
    verbose : str, optional
        Print details on the activities of the function (default is yes)

    Returns
    -------
    df_ingestion : DataFrame
        Pandas dataframe storing the data loaded from the url in config_file_name

    param_config_ingestion : dict
        Dictionary storing the configuration parameters
    """

    import json
    import pandas as pd

    if verbose == 'yes':
        print('data_ingestion: ' + 'starting the data ingestion phase')

    with open(config_file_name) as json_file:  # opening the config_file_name
        param_config_ingestion = json.load(json_file)  # loading the json

        if verbose == 'yes':
            print('data_ingestion: ' + 'json file loading completed!')

    columns_to_read = list(param_config_ingestion['input_parameter']['columns_to_load_from_url'].split(','))
    df_ingestion = pd.read_csv(param_config_ingestion['input_parameter']['source_data_url'],
                               usecols=columns_to_read)
    df_ingestion[param_config_ingestion['input_parameter']['datetime_column_name']] = \
        pd.to_datetime(df_ingestion[param_config_ingestion['input_parameter']['datetime_column_name']],
                       format=param_config_ingestion['input_parameter']['datetime_format'])
    if "" != param_config_ingestion['input_parameter']['index_column_name']:
        df_ingestion.index = df_ingestion[param_config_ingestion['input_parameter']['index_column_name']]
    if verbose == 'yes':
        print('data_ingestion: ' + 'data frame (df) creation completed!')
        print('data_ingestion: summary of statistics *** ')
        print('                |-> number of rows: ' + str(len(df_ingestion)))
        print('                |-> number of columns: ' + str(len(df_ingestion.columns)))
        print('                |-> column names: ' + str(list(df_ingestion.columns)))
        print('                |-> number of missing data: ' + str(list(df_ingestion.isnull().sum())))

    return df_ingestion, param_config_ingestion


def data_selection(data_frame, param_config=None, column_name=None, value=None, column_name_datetime=None,
                   init_datetime=None,
                   end_datetime=None, verbose='yes'):
    """This allows the user to select only a part of a time series in DataFrame, according to some criteria.

    The selected data is returned in a Pandas' DataFrame object.

    Parameters
    ----------
    data_frame : DataFrame
        Pandas dataframe storing the data loaded from the url in config_file_name
    param_config : dict, optional
        A timex json configuration file. If this parameter is present, all the other parameters are
        derived from this one (see json configuration of timex)
    column_name : str, optional
        The column where the selection with 'value' is applied. Default is None
    value : str, optional
        The value for the selection. Default is None
    column_name_datetime : str
        ???
    init_datetime : str, optional
        The first index row of the data_frame to be considered. Default is None
    end_datetime : str, optional
        The last index row of the data_frame to be considered. Default is None
    verbose : str, optional
        Print details on the activities of the function (default is yes).

    Returns
    -------
    df: DataFrame
        Pandas' DataFrame after the selection phase.
    """

    import datetime

    if verbose == 'yes':
        print('data_selection: ' + 'starting the data selection phase')
        print('data_selection: total amount of rows before the selection phase = ' + str(len(data_frame)))

    if param_config:
        column_name = param_config['selection_parameter']['column_name_selection']
        value = param_config['selection_parameter']['value_selection']
        column_name_datetime = param_config['input_parameter']['datetime_column_name']
        init_datetime = datetime.datetime.strptime(param_config['selection_parameter']['init_datetime'], param_config['input_parameter']['datetime_format'])
        end_datetime = datetime.datetime.strptime(param_config['selection_parameter']['end_datetime'], param_config['input_parameter']['datetime_format'])

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


def data_description(data_frame, param_config, verbose='yes'):
    """Function which describes the data stored in data_frame.

    Parameters
    ----------
    data_frame : DataFrame
        Pandas' dataframe containing the data to describe
    param_config : dict
        Dictionary with the configuration parameters
    verbose : str, optional
        Print details on the activities of the function (default is yes)
    """

    if verbose == 'yes':
        print('data_description: ' + 'starting the description of the data')

    data_frame_visualization(data_frame, param_config, verbose=verbose)


def data_frame_visualization(data_frame, param_config, visualization_type="line", mode="independent", verbose="yes"):
    """Function for the visualization of time-series stored in data_frame, using plotly.

    Parameters
    ----------
    data_frame : DataFrame
        Pandas' dataframe with the data to visualize
    param_config : dict
        Dictionary with the configuration parameters
    visualization_type : str, optional
        The type of visualization. Can be "line" or "hist". Default is "list"
    mode : str, optional
        Can be "independent" or "stacked". In independent mode one image is created for each time series.
        In "stacked" mode only one image, with all the time series, is created.
    verbose : str, optional
        Print details on the activities of the function (default is yes)
    """

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import os
    from datetime import datetime

    nticks = 10     # Number of ticks in the y axis
    nbinsx = 20     # Number of bins in the histogram

    if verbose == 'yes':
        print('data_visualization: ' + 'starting the description of the data')

    if mode == "independent":
        # Creating the titles for the subplots
        sub_plot_titles = []
        for col in data_frame.columns:
            if col != param_config['input_parameter']['datetime_column_name']:
                sub_plot_titles.append(col)

        fig = make_subplots(rows=len(data_frame.columns)-1, cols=1, subplot_titles=sub_plot_titles)
    elif mode == "stacked":
        fig = go.Figure()
    else:
        print('ERROR: Visualization type NOT recognized')
        exit()

    i = 1
    for col in data_frame.columns:
        if col != param_config['input_parameter']['datetime_column_name']:
            if visualization_type == 'line':
                if mode == "independent":
                    fig.add_trace(go.Scatter(
                        x=data_frame[param_config['input_parameter']['datetime_column_name']],
                        y=data_frame[col],
                        name=col,
                        mode='lines+markers'
                    ), row=i, col=1)
                else:
                    fig.add_trace(go.Scatter(
                        x=data_frame[param_config['input_parameter']['datetime_column_name']],
                        y=data_frame[col],
                        name=col,
                        mode='lines+markers'
                    ))
            elif visualization_type == 'hist':
                fig.add_trace(go.Histogram(
                    x=data_frame[col], nbinsx=nbinsx
                ), row=i, col=1)
            # Do yet another thing
            else:
                print('ERROR: Visualization type NOT recognized')
                exit()
            i = i + 1

    fig.update_yaxes(nticks=nticks)
    fig.show()

    if param_config['output_parameter']['save_to_file_image'] == 'yes':
        if verbose == 'yes':
            print('data_visualization: ' + 'writing images to file')

        path_output_image_folder = str(param_config['output_parameter']['output_directory'] +
                                       '/images/data_visualization')

        if verbose == 'yes':
            print('data_visualization: ' + 'folder: ' + path_output_image_folder)

        if not os.path.exists(path_output_image_folder):
            os.makedirs(path_output_image_folder)

        dateTimeObj = datetime.now()
        # get the date object from datetime object
        date = str(dateTimeObj.date())

        for col in data_frame.columns:
            if col != param_config['input_parameter']['datetime_column_name']:
                fig = go.Figure()
                fig.update_layout(
                    title={
                        'text': col,
                        'y': 0.9,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'})
                if visualization_type == 'line':
                    fig.add_trace(go.Scatter(
                        x=data_frame[param_config['input_parameter']['datetime_column_name']],
                        y=data_frame[col],
                        mode='lines+markers'))
                elif visualization_type == 'hist':
                    fig.add_trace(go.Histogram(
                        x = data_frame[col], nbinsx=nbinsx))

                fig.write_image(path_output_image_folder + "/" + "data_vis_" + col + "_" + visualization_type + "_" + date + ".png")
                fig.write_image(path_output_image_folder + "/" + "data_vis_" + col + "_" + visualization_type + "_"+ date + ".pdf")

