import webbrowser
from threading import Timer

from pandas import Grouper
import plotly.graph_objects as go

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px

from src.scenario.scenario import Scenario


def data_description_new(scenarios: [Scenario], param_config: dict):
    """
    Function which describes the data stored in each scenario, in a Dash application.
    A scenario is a time series of interest; in a univariate environment,
    each column of a starting dataframe corresponds to a different scenario.

    Parameters
    ----------
    scenarios : [Scenario]
        List of scenario objects.
    param_config : dict
        Dictionary with the configuration parameters
    """
    if param_config["verbose"] == 'yes':
        print('-----------------------------------------------------------')
        print('Data_description: ' + 'starting the description of the data')

    visualization_parameters = param_config["visualization_parameters"]

    # Initialize Dash app.
    app = dash.Dash(__name__)

    children = [
        html.H1(children=param_config["activity_title"]),
    ]

    for s in scenarios:
        ingested_data = s.ingested_data
        model_results = s.model_results

        name = ingested_data.columns[0]

        children.extend([
            html.H2(children=name + " analysis"),
            html.H3("Data visualization")
        ])

        # Plots
        children.extend([
            # Simple plot
            dcc.Graph(
                figure=go.Figure(data=go.Scatter(x=ingested_data.index, y=ingested_data.iloc[:, 0], mode='lines+markers'))
            ),
            # Histogram
            dcc.Graph(
                figure=px.histogram(ingested_data, x=name)
            )
        ])

        # Box plot
        temp = ingested_data[name]
        groups = temp.groupby(Grouper(freq=visualization_parameters["box_plot_frequency"]))

        boxes = []
        for group in groups:
            boxes.append(go.Box(
                name=str(group[0]),
                y=group[1]
            ))

        children.append(
            dcc.Graph(
                figure=go.Figure(data=boxes)
            )
        )

        # Add prediction results
        children.append(
            html.H3("Training & Prediction results"),
        )

        for model in model_results:
            predicted_data = model.prediction
            training_performance = model.training_performance.get_dict()
            model_characteristic = model.characteristics

            children.extend([
                html.Div("Model characteristics:"),
                html.Ul([html.Li(key + ": " + str(model_characteristic[key])) for key in model_characteristic])
            ])
            children.extend([
                html.Div("Training performance:"),
                html.Ul([html.Li(key + ": " + str(training_performance[key])) for key in training_performance])
            ])

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=predicted_data.index, y=predicted_data['yhat'],
                                     mode='lines+markers',
                                     name='yhat'))
            fig.add_trace(go.Scatter(x=predicted_data.index, y=predicted_data['yhat_lower'],
                                     line=dict(color='lightgreen', dash='dash'),
                                     name='yhat_lower'))
            fig.add_trace(go.Scatter(x=predicted_data.index, y=predicted_data['yhat_upper'],
                                     line=dict(color='lightgreen', dash='dash'),
                                     name='yhat_upper'))

            test_percentage = param_config['model_parameters']['test_percentage']
            test_values = int(round(len(ingested_data) * (test_percentage / 100)))

            fig.add_trace(go.Scatter(x=predicted_data.index, y=ingested_data.iloc[:-test_values, 0],
                                     line=dict(color='black'),
                                     mode='markers',
                                     name='training data'))

            fig.add_trace(go.Scatter(x=ingested_data.iloc[-test_values:].index, y=ingested_data.iloc[-test_values:, 0],
                                     line=dict(color='red'),
                                     mode='markers',
                                     name='test data'))
            children.append(
                dcc.Graph(
                    figure=fig
                )
            )

    # That's all. Launch the app.
    app.layout = html.Div(children=children)

    def open_browser():
        webbrowser.open("http://127.0.0.1:8050")

    Timer(1, open_browser).start()
    app.run_server(debug=True, use_reloader=False)


# def data_description(ingested_data, prediction_data, training_performance, param_config):
#     """Function which describes the data stored in data_frame.
#
#     Parameters
#     ----------
#     prediction_data
#     ingested_data : DataFrame
#         Pandas' dataframe containing the data to describe
#     param_config : dict
#         Dictionary with the configuration parameters
#     """
#     if param_config["verbose"] == 'yes':
#         print('-----------------------------------------------------------')
#         print('Data_description: ' + 'starting the description of the data')
#
#     # data_frame_visualization_plotly(data_frame, param_config)
#     data_frame_visualization_dash(ingested_data, prediction_data, training_performance, param_config)
#
#
# def data_frame_visualization_dash(ingested_data, predicted_data, training_performance, param_config,
#                                   mode='independent'):
#     visualization_parameters = param_config["visualization_parameters"]
#
#     app = dash.Dash(__name__)
#
#     children = [
#         html.H1(children=param_config["activity_title"]),
#         html.H2("Data visualization")
#     ]
#
#     if mode == "independent":
#         for col in ingested_data.columns:
#             children.append(
#                 html.Div(children=col + " analysis")
#             )
#             # Simple plot
#             children.append(
#                 dcc.Graph(
#                     figure=go.Figure(data=go.Scatter(x=ingested_data.index, y=ingested_data[col], mode='lines+markers'))
#                 ))
#             # Histogram
#             children.append(
#                 dcc.Graph(
#                     figure=px.histogram(ingested_data, x=col)
#                 )
#             )
#             # Box plot
#             temp = ingested_data[col]
#             groups = temp.groupby(Grouper(freq=visualization_parameters["box_plot_frequency"]))
#
#             boxes = []
#             for group in groups:
#                 boxes.append(go.Box(
#                     name=str(group[0]),
#                     y=group[1]
#                 ))
#
#             children.append(
#                 dcc.Graph(
#                     figure=go.Figure(data=boxes)
#                 )
#             )
#
#             # Autocorrelation plot
#             # children.append(
#             #     dcc.Graph(
#             #         figure=go.Figure(pd.plotting.autocorrelation_plot(data_frame[col]))
#             #     )
#             # )
#
#     elif mode == "stacked":
#         fig = go.Figure()
#         for col in ingested_data.columns:
#             fig.add_trace(go.Scatter(x=ingested_data.index, y=ingested_data[col], mode='lines+markers', name=col))
#
#         children.append(
#             html.Div(children="stacked"),
#         )
#         children.append(
#             dcc.Graph(
#                 figure=fig)
#         )
#
#     # Add prediction results
#     children.append(
#         html.H2("Training & Prediction results")
#     )
#
#     children.append(
#         html.Div("Training performance:")
#     )
#     children.append(
#         html.Ul([html.Li(key + ": " + str(value)) for key, value in training_performance.items()])
#     )
#
#     fig = go.Figure()
#
#     fig.add_trace(go.Scatter(x=predicted_data.index, y=predicted_data['yhat'],
#                              mode='lines+markers',
#                              name='yhat'))
#     fig.add_trace(go.Scatter(x=predicted_data.index, y=predicted_data['yhat_lower'],
#                              line=dict(color='lightgreen', dash='dash'),
#                              name='yhat_lower'))
#     fig.add_trace(go.Scatter(x=predicted_data.index, y=predicted_data['yhat_upper'],
#                              line=dict(color='lightgreen', dash='dash'),
#                              name='yhat_upper'))
#
#     test_percentage = param_config['model_parameters']['test_percentage']
#     test_values = int(round(len(ingested_data) * (test_percentage / 100)))
#
#     fig.add_trace(go.Scatter(x=predicted_data.index, y=ingested_data.iloc[:-test_values, 0],
#                              line=dict(color='black'),
#                              mode='markers',
#                              name='training data'))
#
#     fig.add_trace(go.Scatter(x=ingested_data.iloc[-test_values:].index, y=ingested_data.iloc[-test_values:, 0],
#                              line=dict(color='red'),
#                              mode='markers',
#                              name='test data'))
#     children.append(
#         dcc.Graph(
#             figure=fig
#         )
#     )
#
#     app.layout = html.Div(children=children)
#
#     def open_browser():
#         webbrowser.open("http://127.0.0.1:8050")
#
#     Timer(1, open_browser).start()
#     app.run_server(debug=True, use_reloader=False)
#
#
# def data_frame_visualization_plotly(data_frame, param_config, visualization_type="line", mode="independent"):
#     """Function for the visualization of time-series stored in data_frame, using plotly.
#
#     Parameters
#     ----------
#     data_frame : DataFrame
#         Pandas' dataframe with the data to visualize
#     param_config : dict
#         Dictionary with the configuration parameters
#     visualization_type : str, optional
#         The type of visualization. Can be "line" or "hist". Default is "line"
#     mode : str, optional
#         Can be "independent" or "stacked". In independent mode one image is created for each time series.
#         In "stacked" mode only one image, with all the time series, is created.
#     """
#     nticks = 10  # Number of ticks in the y axis
#     nbinsx = 20  # Number of bins in the histogram
#
#     verbose = param_config["verbose"]
#     datetime_column_name = param_config["input_parameters"]["datetime_column_name"]
#
#     data_frame = data_frame.reset_index()
#     if verbose == 'yes':
#         print('data_visualization: ' + 'starting the description of the data')
#
#     if mode == "independent":
#         # Creating the titles for the subplots
#         sub_plot_titles = []
#         for col in data_frame.columns:
#             if col != datetime_column_name:
#                 sub_plot_titles.append(col)
#
#         fig = make_subplots(rows=len(data_frame.columns) - 1, cols=1, subplot_titles=sub_plot_titles)
#
#     elif mode == "stacked":
#         fig = go.Figure()
#
#     else:
#         print('ERROR: Visualization type NOT recognized')
#         exit()
#
#     i = 1
#
#     for col in data_frame.columns:
#         if col != datetime_column_name:
#             if visualization_type == 'line':
#                 if mode == "independent":
#                     fig.add_trace(go.Scatter(
#                         x=data_frame[datetime_column_name],
#                         y=data_frame[col],
#                         name=col,
#                         mode='lines+markers'
#                     ), row=i, col=1)
#                 else:
#                     fig.add_trace(go.Scatter(
#                         x=data_frame[datetime_column_name],
#                         y=data_frame[col],
#                         name=col,
#                         mode='lines+markers'
#                     ))
#             elif visualization_type == 'hist':
#                 fig.add_trace(go.Histogram(
#                     x=data_frame[col], nbinsx=nbinsx
#                 ), row=i, col=1)
#             # Do yet another thing
#             else:
#                 print('ERROR: Visualization type NOT recognized')
#                 exit()
#             i = i + 1
#
#     fig.update_yaxes(nticks=nticks)
#     fig.show()
#
#     if param_config['output_parameters']['save_to_file_image'] == 'yes':
#         if verbose == 'yes':
#             print('data_visualization: ' + 'writing images to file')
#
#         path_output_image_folder = str(param_config['output_parameters']['output_directory'] +
#                                        '/images/data_visualization')
#
#         if verbose == 'yes':
#             print('data_visualization: ' + 'folder: ' + path_output_image_folder)
#
#         if not os.path.exists(path_output_image_folder):
#             os.makedirs(path_output_image_folder)
#
#         dateTimeObj = datetime.now()
#         # get the date object from datetime object
#         date = str(dateTimeObj.date())
#
#         for col in data_frame.columns:
#             if col != datetime_column_name:
#                 fig = go.Figure()
#                 fig.update_layout(
#                     title={
#                         'text': col,
#                         'y': 0.9,
#                         'x': 0.5,
#                         'xanchor': 'center',
#                         'yanchor': 'top'})
#                 if visualization_type == 'line':
#                     fig.add_trace(go.Scatter(
#                         x=data_frame[datetime_column_name],
#                         y=data_frame[col],
#                         mode='lines+markers'))
#                 elif visualization_type == 'hist':
#                     fig.add_trace(go.Histogram(
#                         x=data_frame[col], nbinsx=nbinsx))
#
#                 fig.write_image(
#                     path_output_image_folder + "/" + "data_vis_" + col + "_" + visualization_type + "_" + date + ".png")
#                 fig.write_image(
#                     path_output_image_folder + "/" + "data_vis_" + col + "_" + visualization_type + "_" + date + ".pdf")
