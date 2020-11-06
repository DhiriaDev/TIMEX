import webbrowser
from threading import Timer

import pandas
from pandas import Grouper, DataFrame
import plotly.graph_objects as go
import numpy as np


import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from plotly.subplots import make_subplots


from timex.data_prediction.data_prediction import TestingPerformance, SingleResult
from timex.scenario.scenario import Scenario


def create_dash_children(scenarios: [Scenario], param_config: dict):
    children = []
    children.append(
        html.Div(
            html.P(['The data contained in this page is an example of the capabilities of TIMEX, '
                    'a work-in-progress framework for automatic time series analysis.', html.Br(),
                    'The forecasts provided in this page are relative to the evolution of Covid-19 in Italy, '
                    'built with the ', html.A("data", href='https://github.com/pcm-dpc/COVID-19', target="_blank"),
                    ' provided by Italian Protezione Civile.', html.Br(), html.Br(),
                    'The information on this site is not intended or implied to be a substitute for '
                    'professional medical advice, diagnosis or treatment. All content, including text, '
                    'graphics, images and information, contained on or available through this web site is for '
                    'general information purposes only.', html.Br(), 'We make no representation and assume no '
                    'responsibility for the accuracy of information contained on or available through this web '
                    'site, and such information is subject to change without notice. You are encouraged to '
                    'confirm any information obtained from or through this web site with other sources.',
                    html.Br(),
                    html.Br(),
                    'For suggestions and questions contact us at manuel.roveri (at) polimi.it or alessandro.falcetta '
                    '(at) mail.polimi.it '
                    ])))

    visualization_parameters = param_config["visualization_parameters"]
    model_parameters = param_config["model_parameters"]

    for s in scenarios:
        ingested_data = s.ingested_data
        models = s.models

        # Data visualization with plots
        children.extend([
            html.H1(children=param_config["activity_title"]),
            html.H2(children=ingested_data.columns[0] + " analysis"),
            html.H3("Data visualization"),
            line_plot(ingested_data),
            histogram_plot(ingested_data),
            box_plot(ingested_data, visualization_parameters["box_plot_frequency"]),
            autocorrelation_plot(ingested_data)
        ])

        # Prediction results
        children.append(
            html.H3("Training & Prediction results"),
        )

        for model in models:
            model_results = model.results
            model_characteristic = model.characteristics

            test_percentage = model_parameters['test_percentage']
            test_values = int(round(len(ingested_data) * (test_percentage / 100)))

            main_accuracy_estimator = model_parameters["main_accuracy_estimator"]
            model_results.sort(key=lambda x: getattr(x.testing_performances, main_accuracy_estimator.upper()))

            best_prediction = model_results[0].prediction
            testing_performances = [x.testing_performances for x in model_results]

            children.extend([
                html.Div("Model characteristics:"),
                html.Ul([html.Li(key + ": " + str(model_characteristic[key])) for key in model_characteristic]),
                # html.Div("Testing performance:"),
                # html.Ul([html.Li(key + ": " + str(testing_performances[key])) for key in testing_performances]),
                prediction_plot(ingested_data, best_prediction, test_values),
                performance_plot(ingested_data, best_prediction, testing_performances, test_values),
            ])

            # EXTRA
            # Warning: this will plot every model result, with every training set used!
            # children.extend(plot_every_prediction(ingested_data, model_results, main_accuracy_estimator, test_values))

    return children


def data_description_new(scenarios: [Scenario], param_config: dict):
    """
    Function which describes the data stored in each scenario, in a Dash application.
    A scenario is a time series of interest; in a univariate environment, each column
    of a starting dataframe corresponds to a different scenario.

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
    model_parameters = param_config["model_parameters"]

    # Initialize Dash app.
    app = dash.Dash(__name__)

    children = [
        html.H1(children=param_config["activity_title"]),
    ]

    for s in scenarios:
        ingested_data = s.ingested_data
        models = s.models

        # Data visualization with plots
        children.extend([
            html.H2(children=ingested_data.columns[0] + " analysis"),
            html.H3("Data visualization"),
            line_plot(ingested_data),
            histogram_plot(ingested_data),
            box_plot(ingested_data, visualization_parameters["box_plot_frequency"]),
            autocorrelation_plot(ingested_data)
        ])

        # Prediction results
        children.append(
            html.H3("Training & Prediction results"),
        )

        for model in models:
            model_results = model.results
            model_characteristic = model.characteristics

            test_percentage = model_parameters['test_percentage']
            test_values = int(round(len(ingested_data) * (test_percentage / 100)))

            main_accuracy_estimator = model_parameters["main_accuracy_estimator"]
            model_results.sort(key=lambda x: getattr(x.testing_performances, main_accuracy_estimator.upper()))

            best_prediction = model_results[0].prediction
            testing_performances = [x.testing_performances for x in model_results]

            children.extend([
                html.Div("Model characteristics:"),
                html.Ul([html.Li(key + ": " + str(model_characteristic[key])) for key in model_characteristic]),
                # html.Div("Testing performance:"),
                # html.Ul([html.Li(key + ": " + str(testing_performances[key])) for key in testing_performances]),
                prediction_plot(ingested_data, best_prediction, test_values),
                performance_plot(ingested_data, best_prediction, testing_performances, test_values),
            ])

            # EXTRA
            # Warning: this will plot every model result, with every training set used!
            # children.extend(plot_every_prediction(ingested_data, model_results, main_accuracy_estimator, test_values))

    # That's all. Launch the app.
    app.layout = html.Div(children=children)

    def open_browser():
        webbrowser.open("http://127.0.0.1:8050")

    Timer(1, open_browser).start()
    app.run_server(debug=True, use_reloader=False)


def line_plot(ingested_data: DataFrame) -> dcc.Graph:
    fig = go.Figure(data=go.Scatter(x=ingested_data.index, y=ingested_data.iloc[:, 0], mode='lines+markers'))
    fig.update_layout(title='Line plot', xaxis_title=ingested_data.index.name, yaxis_title=ingested_data.columns[0])

    g = dcc.Graph(
        figure=fig
    )
    return g


def histogram_plot(ingested_data: DataFrame) -> dcc.Graph:
    fig = px.histogram(ingested_data, ingested_data.columns[0])
    fig.update_layout(title='Histogram')
    g = dcc.Graph(
        figure=fig
    )
    return g


def autocorrelation_plot(ingested_data: DataFrame) -> dcc.Graph:
    # Code from https://github.com/pandas-dev/pandas/blob/v1.1.4/pandas/plotting/_matplotlib/misc.py
    n = len(ingested_data)
    data = np.asarray(ingested_data)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def r(h):
        return ((data[: n - h] - mean) * (data[h:] - mean)).sum() / float(n) / c0

    x = np.arange(n) + 1
    y = [r(loc) for loc in x]

    z95 = 1.959963984540054
    z99 = 2.5758293035489004

    c1 = z99 / np.sqrt(n)
    c2 = z95 / np.sqrt(n)
    c3 = -z95 / np.sqrt(n)
    c4 = -z99 / np.sqrt(n)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='autocorrelation'))
    fig.add_trace(go.Scatter(x=x, y=np.full(n, c1), line=dict(color='gray', width=1), name='z99'))
    fig.add_trace(go.Scatter(x=x, y=np.full(n, c2), line=dict(color='gray', width=1), name='z95'))
    fig.add_trace(go.Scatter(x=x, y=np.full(n, c3), line=dict(color='gray', width=1), name='-z95'))
    fig.add_trace(go.Scatter(x=x, y=np.full(n, c4), line=dict(color='gray', width=1), name='-z99'))
    fig.update_layout(title='Autocorrelation plot', xaxis_title='Lag', yaxis_title='Autocorrelation')
    fig.update_yaxes(tick0=-1.0, dtick=0.25)
    fig.update_yaxes(range=[-1.2, 1.2])
    g = dcc.Graph(
        figure=fig
    )
    return g


def box_plot(ingested_data: DataFrame, freq: str) -> dcc.Graph:
    temp = ingested_data.iloc[:, 0]
    groups = temp.groupby(Grouper(freq=freq))

    boxes = []

    for group in groups:
        boxes.append(go.Box(
            name=str(group[0]),
            y=group[1]
        ))

    fig = go.Figure(data=boxes)
    fig.update_layout(title='Box plot', xaxis_title=ingested_data.index.name, yaxis_title='Count')

    g = dcc.Graph(
        figure=fig
    )
    return g


def prediction_plot(ingested_data: DataFrame, predicted_data: DataFrame, test_values: int) -> dcc.Graph:
    fig = go.Figure()

    not_training_data = ingested_data.loc[:predicted_data.index[0]]
    training_data = ingested_data.loc[predicted_data.index[0]:]
    training_data = training_data.iloc[:-test_values]
    test_data = ingested_data.iloc[-test_values:]

    fig.add_trace(go.Scatter(x=predicted_data.index, y=predicted_data['yhat'],
                             mode='lines+markers',
                             name='yhat'))
    try:
        fig.add_trace(go.Scatter(x=predicted_data.index, y=predicted_data['yhat_lower'],
                                 line=dict(color='lightgreen', dash='dash'),
                                 name='yhat_lower'))
        fig.add_trace(go.Scatter(x=predicted_data.index, y=predicted_data['yhat_upper'],
                                 line=dict(color='lightgreen', dash='dash'),
                                 name='yhat_upper'))
    except:
        pass

    fig.add_trace(go.Scatter(x=not_training_data.index, y=not_training_data.iloc[:, 0],
                             line=dict(color='black'),
                             mode='markers',
                             name='unused data'))
    fig.add_trace(go.Scatter(x=training_data.index, y=training_data.iloc[:, 0],
                             line=dict(color='green'),
                             mode='markers',
                             name='training data'))

    fig.add_trace(go.Scatter(x=test_data.index, y=test_data.iloc[:, 0],
                             line=dict(color='red'),
                             mode='markers',
                             name='test data'))
    fig.update_layout(title="Best prediction", xaxis_title=ingested_data.index.name, yaxis_title=ingested_data.columns[0])
    g = dcc.Graph(
        figure=fig
    )
    return g


def performance_plot(ingested_data: DataFrame, predicted_data: DataFrame, testing_performances: [TestingPerformance],
                     test_values: int) -> dcc.Graph:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    training_data = ingested_data.iloc[:-test_values]

    data_performances = []

    for tp in testing_performances:
        data_performances.append([tp.first_used_index, tp.MAE, tp.MSE])

    df_performances = pandas.DataFrame(data_performances, columns=['index', 'mae', 'mse'])
    df_performances.set_index('index', drop=True, inplace=True)
    df_performances.sort_index(inplace=True)

    fig.append_trace(go.Scatter(x=df_performances.index, y=df_performances['mae'],
                                line=dict(color='red'),
                                mode="lines+markers",
                                name='MAE'), row=1, col=1)

    fig.append_trace(go.Scatter(x=df_performances.index, y=df_performances['mse'],
                                line=dict(color='green'),
                                mode="lines+markers",
                                name='MSE'), row=2, col=1)
    fig.append_trace(go.Scatter(x=training_data.index, y=training_data.iloc[:, 0],
                                line=dict(color='black'),
                                mode='markers',
                                name='training data'), row=3, col=1)

    # Small trick to make the x-axis have the same length of the "Prediction plot"
    predicted_data.iloc[:, 0] = "nan"
    fig.append_trace(go.Scatter(x=predicted_data.index, y=predicted_data.iloc[:, 0],
                             mode='lines+markers',
                             name='yhat', showlegend=False), row=3, col=1)

    fig.update_yaxes(title_text="MAE", row=1, col=1)
    fig.update_yaxes(title_text="MSE", row=2, col=1)
    fig.update_yaxes(title_text=ingested_data.columns[0], row=3, col=1)

    fig.update_layout(title='Performances with different training windows', height=700)
    g = dcc.Graph(
        figure=fig
    )
    return g


def plot_every_prediction(ingested_data: DataFrame, model_results: [SingleResult],
                          main_accuracy_estimator: str, test_values: int):

    new_childrens = [html.Div("EXTRA: plot _EVERY_ prediction\n")]

    model_results.sort(key=lambda x: len(x.prediction))

    for r in model_results:
        predicted_data = r.prediction
        testing_performance = r.testing_performances
        plot = prediction_plot(ingested_data, predicted_data, test_values)
        plot.figure.update_layout(title="")
        new_childrens.extend([
            html.Div(main_accuracy_estimator.upper()
                     + ": " + str(getattr(testing_performance, main_accuracy_estimator.upper()))),
            plot
        ])

    return new_childrens

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
