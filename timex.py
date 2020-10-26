# LIST OF REQUIRED PACKAGES (to be imported):
#       - pandas
#       - numpy
#       - plotly
#       - psutil
#       - kaleido

def data_ingestion(config_file_name, verbose='yes'):
    # function for the ingestion of data
    #   Input Parameters:
    #       - config_file_name: filename of the json config file
    #       - verbose (yes/no): printing the activities of the function
    #
    #   Output Parameters:
    #       - df: Pandas dataframe storing the data loaded from the url in config_file_name
    #       - config_file_name: the dictionary storing the configuration parameters
    #
    # timex package - Politecnico di Milano
    # v 1.0 - October 2020

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
    # function for the selection of data
    #   Input Parameters:
    #       - data_frame: Pandas dataframe storing the data loaded from the url in config_file_name
    #       - param_config: a timex json configuration file. If this parameter is present, all the other parameters are
    #                       derived from this one (see json configuration of timex
    #       - column_name (default = ''): the column where the selection with 'value' is applied
    #       - value  (default = ''): the value for the selection
    #       - init_datetime (default = ''): the first index row of the data_frame to be considered
    #       - end_datetime (default = ''): the last index row of the data_frame to be considered
    #       - verbose (yes/no): printing the activities of the function
    #
    #   Output Parameters:
    #       - df: Pandas dataframe after the selection phase
    #
    # timex package - Politecnico di Milano
    # v 1.0 - October 2020

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
    # Function for the adding a 1-step diff column computed on the  column_name_target_diff of the data frame.
    # The function automatically removes the first row of the data_frame since the diff value is nan
    # If the name_diff_column parameter is specified, it is used as name of the new column.
    # Otherwise, the name 'diff' is used
    #
    #   Input Parameters:
    #       - data_frame: Pandas dataframe storing the data loaded from the url in config_file_name
    #       - column_name (default = ''): the column where the selection with 'value' is applied
    #       - verbose (yes/no): printing the activities of the function
    #
    #   Output Parameters:
    #       - df: Pandas dataframe with the new diff column and without the first row
    #
    # timex package - Politecnico di Milano
    # v 1.0 - October 2020

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
    # Function for description of the data stored in data_frame.
    # This function activates specific functions:
    #       - data_visualization
    #
    #   Input Parameters:
    #       - data_frame: Pandas dataframe storing the data loaded from the url in config_file_name
    #       - config_file_name: the dictionary storing the configuration parameters
    #       - verbose (yes/no): printing the activities of the function
    #
    #
    # timex package - Politecnico di Milano
    # v 1.0 - October 2020

    if verbose == 'yes':
        print('data_description: ' + 'starting the description of the data')

    data_frame_visualization(data_frame, param_config, verbose=verbose)


def data_frame_visualization(data_frame, param_config, visualization_type="line", mode="independent", verbose="yes"):
    # Function for the visualization of time-series stored in data_frame.
    # If mode="independent", one image is created for each time-series. Otherwise, when mode="stacked", all the time-series
    # are presented in a single image.
    # This function relies on plotly Python package.
    #
    #   Input Parameters:
    #       - data_frame: Pandas dataframe storing the data loaded from the url in config_file_name
    #       - param_config (default = ''): the dictionary storing the configuration parameters
    #       - visualization_type (default = 'line'): the type of visualization.
    #                                   Available types: line, hist
    #       - mode (default = "independent"): the way time-series are plotted.
    #                   Independent: one image for time-series; Stacked: one image for all the time-series
    #       - verbose (yes/no): printing the activities of the function
    #
    #
    # timex package - Politecnico di Milano
    # v 1.0 - October 2020

    # Dependencies:
    #   - kaleido

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

