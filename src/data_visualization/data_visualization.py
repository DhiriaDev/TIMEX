from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
from datetime import datetime


def data_description(data_frame, param_config):
    """Function which describes the data stored in data_frame.

    Parameters
    ----------
    data_frame : DataFrame
        Pandas' dataframe containing the data to describe
    param_config : dict
        Dictionary with the configuration parameters
    """
    if param_config["verbose"] == 'yes':
        print('data_description: ' + 'starting the description of the data')

    data_frame_visualization(data_frame, param_config)


def data_frame_visualization(data_frame, param_config, visualization_type="line", mode="independent"):
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
    """
    nticks = 10  # Number of ticks in the y axis
    nbinsx = 20  # Number of bins in the histogram

    verbose = param_config["verbose"]
    datetime_column_name = param_config["input_parameters"]["datetime_column_name"]

    if verbose == 'yes':
        print('data_visualization: ' + 'starting the description of the data')

    if mode == "independent":
        # Creating the titles for the subplots
        sub_plot_titles = []
        for col in data_frame.columns:
            if col != datetime_column_name:
                sub_plot_titles.append(col)

        fig = make_subplots(rows=len(data_frame.columns) - 1, cols=1, subplot_titles=sub_plot_titles)

    elif mode == "stacked":
        fig = go.Figure()

    else:
        print('ERROR: Visualization type NOT recognized')
        exit()

    i = 1
    for col in data_frame.columns:
        if col != datetime_column_name:
            if visualization_type == 'line':
                if mode == "independent":
                    fig.add_trace(go.Scatter(
                        x=data_frame[datetime_column_name],
                        y=data_frame[col],
                        name=col,
                        mode='lines+markers'
                    ), row=i, col=1)
                else:
                    fig.add_trace(go.Scatter(
                        x=data_frame[datetime_column_name],
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

    if param_config['output_parameters']['save_to_file_image'] == 'yes':
        if verbose == 'yes':
            print('data_visualization: ' + 'writing images to file')

        path_output_image_folder = str(param_config['output_parameters']['output_directory'] +
                                       '/images/data_visualization')

        if verbose == 'yes':
            print('data_visualization: ' + 'folder: ' + path_output_image_folder)

        if not os.path.exists(path_output_image_folder):
            os.makedirs(path_output_image_folder)

        dateTimeObj = datetime.now()
        # get the date object from datetime object
        date = str(dateTimeObj.date())

        for col in data_frame.columns:
            if col != datetime_column_name:
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
                        x=data_frame[datetime_column_name],
                        y=data_frame[col],
                        mode='lines+markers'))
                elif visualization_type == 'hist':
                    fig.add_trace(go.Histogram(
                        x=data_frame[col], nbinsx=nbinsx))

                fig.write_image(
                    path_output_image_folder + "/" + "data_vis_" + col + "_" + visualization_type + "_" + date + ".png")
                fig.write_image(
                    path_output_image_folder + "/" + "data_vis_" + col + "_" + visualization_type + "_" + date + ".pdf")
