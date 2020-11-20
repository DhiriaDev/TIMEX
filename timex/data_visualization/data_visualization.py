import pandas
from pandas import Grouper, DataFrame
import plotly.graph_objects as go
import numpy as np

import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx

from timex.data_prediction.data_prediction import TestingPerformance, SingleResult
from timex.scenario.scenario import Scenario


def create_scenario_children(scenario: Scenario, param_config: dict, xcorr_plot: bool = True):
    """
    Creates the Dash children for a specific scenario. They include a line plot,
    histogram, box plot and autocorrelation plot. For each model on the scenario
    the prediction plot and performance plot are also added.
    
    Parameters
    ----------
    scenario: Scenario
    
    param_config : dict
    
    xcorr_plot : bool
    True to display the cross-correlation plot. Default True.

    Returns
    -------
    List of Dash children.
    """
    children = []

    visualization_parameters = param_config["visualization_parameters"]
    model_parameters = param_config["model_parameters"]

    scenario_data = scenario.scenario_data
    models = scenario.models
    name = scenario_data.columns[0]

    xcorr = calc_xcorr(name, scenario.ingested_data, visualization_parameters["xcorr_max_lags"]) if xcorr_plot else None

    # Data visualization with plots
    children.extend([
        html.H2(children=name + " analysis", id=name),
        html.H3("Data visualization"),
        line_plot(scenario_data),
        histogram_plot(scenario_data),
        box_plot(scenario_data, visualization_parameters["box_plot_frequency"]),
        autocorrelation_plot(scenario_data),
        cross_correlation_plot(xcorr, visualization_parameters["xcorr_max_lags"]) if xcorr_plot else None,
        cross_correlation_graph(name, xcorr, visualization_parameters["xcorr_graph_threshold"]) if xcorr_plot else None
    ])

    # Prediction results
    children.append(
        html.H3("Training & Prediction results"),
    )

    for model in models:
        model_results = model.results
        model_characteristic = model.characteristics

        test_values = model_characteristic["test_values"]
        main_accuracy_estimator = model_parameters["main_accuracy_estimator"]
        model_results.sort(key=lambda x: getattr(x.testing_performances, main_accuracy_estimator.upper()))

        best_prediction = model_results[0].prediction
        testing_performances = [x.testing_performances for x in model_results]

        children.extend([
            characteristics_list(model_characteristic, testing_performances),
            # html.Div("Testing performance:"),
            # html.Ul([html.Li(key + ": " + str(testing_performances[key])) for key in testing_performances]),
            prediction_plot(scenario_data, best_prediction, test_values),
            performance_plot(scenario_data, best_prediction, testing_performances, test_values),
        ])

        # EXTRA
        # Warning: this will plot every model result, with every training set used!
        # children.extend(plot_every_prediction(ingested_data, model_results, main_accuracy_estimator, test_values))

    return children


def create_dash_children(scenarios: [Scenario], param_config: dict):
    """
    Create Dash children, in order, for a list of Scenarios.
    Parameters
    ----------
    scenarios : [Scenario]

    param_config : dict

    Returns
    -------
    List of Dash children.

    """
    children = []
    for s in scenarios:
        children.extend(create_scenario_children(s, param_config))

    return children


def line_plot(df: DataFrame) -> dcc.Graph:
    """
    Create and return the line plot for a dataframe.

    Parameters
    ----------
    df : DataFrame
    Dataframe to plot.

    Returns
    -------
    g : dcc.Graph
    """
    fig = go.Figure(data=go.Scatter(x=df.index, y=df.iloc[:, 0], mode='lines+markers'))
    fig.update_layout(title='Line plot', xaxis_title=df.index.name, yaxis_title=df.columns[0])

    g = dcc.Graph(
        figure=fig
    )
    return g


def line_plot_multiIndex(df: DataFrame) -> dcc.Graph:
    """
    Returns a line plot for a dataframe with a MultiIndex.
    It is assumed that the first-level index is the real index,
    and that data should be grouped using the second-level one.

    Parameters
    ----------
    df : DataFrame
    Dataframe to plot. It is a multiIndex dataframe.

    Returns
    -------
    g : dcc.Graph
    """
    fig = go.Figure()
    for region in df.index.get_level_values(1).unique():
        fig.add_trace(go.Scatter(x=df.index.get_level_values(0).unique(), y=df.loc[
            (df.index.get_level_values(1) == region), df.columns[0]], name=region))

    fig.update_layout(title='Line plot', xaxis_title=df.index.get_level_values(0).name,
                      yaxis_title=df.columns[0])
    g = dcc.Graph(
        figure=fig
    )
    return g


def histogram_plot(df: DataFrame) -> dcc.Graph:
    """
    Create and return the histogram plot for a dataframe.

    Parameters
    ----------
    df : DataFrame
    Dataframe to plot.

    Returns
    -------
    g : dcc.Graph
    """

    fig = px.histogram(df, df.columns[0])
    fig.update_layout(title='Histogram')
    g = dcc.Graph(
        figure=fig
    )
    return g


def autocorrelation_plot(df: DataFrame) -> dcc.Graph:
    """
    Create and return the autocorrelation plot for a dataframe.

    Parameters
    ----------
    df : DataFrame
    Dataframe to use in the autocorrelation plot.

    Returns
    -------
    g : dcc.Graph
    """

    # Code from https://github.com/pandas-dev/pandas/blob/v1.1.4/pandas/plotting/_matplotlib/misc.py
    n = len(df)
    data = np.asarray(df)
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


def calc_xcorr(target: str, ingested_data: DataFrame, lags: int) -> DataFrame:
    """
    Calculate the cross-correlation for the ingested data.
    Use the scenario column as target; the correlation is computed against
    all lags of all other columns which include numbers.

    Parameters
    ----------
    target : str
    Column which should be used as target for the cross correlation.

    ingested_data : DataFrame
    Entire dataframe parsed from app

    lags : int
    Limit the analysis to max lags.

    Returns
    -------
    result : DataFrame
    DataFrame having the lags as index and the correlation value for each column.
    """
    def df_shifted(df, _target=None, lag=0):
        if not lag and not _target:
            return df
        new = {}
        for c in df.columns:
            if c == _target:
                new[c] = df[_target]
            else:
                new[c] = df[c].shift(periods=lag)
        return pandas.DataFrame(data=new)

    columns = ingested_data.columns.tolist()
    columns = [elem for elem in columns if ingested_data[elem].dtype != str and elem != target]

    result = DataFrame(columns=columns)

    for i in range(-lags, lags):
        shifted = df_shifted(ingested_data, target, i)
        if i <= 0:
            shifted = shifted.iloc[:-lags]
        else:
            shifted = shifted.iloc[lags:]

        corr = [shifted[target].corr(other=shifted[col]) for col in columns]
        result.loc[i] = corr

    return result


def cross_correlation_plot(xcorr: DataFrame, lags: int):
    """
    Create and return the cross-correlation plot for all the columns in the dataframe.
    The scenario column is used as target; the correlation is computed against all
    lags of all the other columns which include numbers.


    Parameters
    ----------
    xcorr : DataFrame
    Cross-correlation values.

    lags : int
    Max number of lags to plot.

    Returns
    -------
    g : dcc.Graph
    """

    fig = go.Figure()

    for col in xcorr.columns:
        fig.add_trace(go.Scatter(x=xcorr.index, y=xcorr[col],
                                 mode='lines',
                                 name=col))

    # Formula from https://support.minitab.com/en-us/minitab/18/help-and-how-to/modeling-statistics/time-series/how-to/cross-correlation/interpret-the-results/all-statistics-and-graphs/
    significance_level = DataFrame(columns=['Value'], dtype=np.float64)
    for i in range(-lags, lags):
        significance_level.loc[i] = 2 / np.sqrt(lags - abs(i))

    fig.add_trace(
        go.Scatter(x=significance_level.index, y=significance_level['Value'], line=dict(color='gray', width=1), name='z95'))
    fig.add_trace(
        go.Scatter(x=significance_level.index, y=-significance_level['Value'], line=dict(color='gray', width=1), name='-z95'))

    fig.update_layout(title="Cross-correlation (Pearson)<br><sub>"
                            "Negative lags (left part) show the correlation between this scenario and the future of the others.<br>"
                            "Meanwhile, positive lags (right part) shows the correlation between this scenario and the past of the others.<br>"
                            "</sub>", xaxis_title="Lags",
                      yaxis_title="Correlation")
    fig.update_yaxes(tick0=-1.0, dtick=0.25)
    fig.update_yaxes(range=[-1.2, 1.2])

    g = dcc.Graph(
        figure=fig
    )
    return g


def cross_correlation_graph(name: str, xcorr: DataFrame, threshold: int) -> dcc.Graph:
    """
    Create and return the cross-correlation graph for all the columns in the dataframe.
    The scenario column is used as target; the correlation is computed against all
    lags of all the other columns which include numbers.

    Correlation can be computed with algorithms ‘pearson’, ‘kendall’, ‘spearman'.

    Parameters
    ----------
    name : str
    Name of the target.

    xcorr : DataFrame
    Cross-correlation dataframe.

    threshold : int
    Minimum value of correlation for which a edge should be drawn.

    Returns
    -------
    g : dcc.Graph
    """
    G = nx.DiGraph()
    G.add_nodes_from(xcorr.columns)
    G.add_node(name)

    for col in xcorr.columns:
        index_of_max = xcorr[col].idxmax()
        corr = xcorr.loc[index_of_max, col]
        if abs(corr) > threshold:
            G.add_edge(name, col, corr=corr, lag=index_of_max)

    pos = nx.layout.spring_layout(G)

    # Create Edges
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(color='black'),
        mode='lines',
        hoverinfo='skip',
    )

    for edge in G.edges():
        start = edge[0]
        end = edge[1]
        x0, y0 = pos.get(start)
        x1, y1 = pos.get(end)
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    # Create Nodes
    node_trace = go.Scatter(
        x=[],
        y=[],
        mode='markers+text',
        text=[node for node in G.nodes],
        textposition="bottom center",
        hoverinfo='skip',
        marker=dict(
            color='green',
            size=15)
    )

    for node in G.nodes():
        x, y = pos.get(node)
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

    # Annotations to support arrows
    edges_positions = [e for e in G.edges]
    annotateArrows = [dict(showarrow=True, arrowsize=1.0, arrowwidth=2, arrowhead=2, standoff=2, startstandoff=2,
                           ax=pos[arrow[0]][0], ay=pos[arrow[0]][1], axref='x', ayref='y',
                           x=pos[arrow[1]][0], y=pos[arrow[1]][1], xref='x', yref='y',
                           text="bla") for arrow in edges_positions]

    fig = go.Figure(data=[node_trace, edge_trace],
                    layout=go.Layout(title="Cross-correlation graph",
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     showlegend=False,
                                     annotations=annotateArrows,
                                     height=500))

    # Add annotations on edges
    for e in G.edges:
        lag = str(G.edges[e]['lag'])
        corr = str(round(G.edges[e]['corr'], 3))

        end = e[1]
        x, y = pos.get(end)

        fig.add_annotation(x=x, y=y, text="Lag: " + lag + ", corr: " + corr, yshift=20, showarrow=False,
                           bgcolor='white')

    g = dcc.Graph(
        figure=fig
    )
    return g


def box_plot(df: DataFrame, freq: str) -> dcc.Graph:
    """
    Create and return the box plot for a dataframe.

    Parameters
    ----------
    df : DataFrame
    Dataframe to use in the box plot.

    freq : str
    Frequency which should be used to group the data and
    create the boxes.

    Returns
    -------
    g : dcc.Graph
    """
    temp = df.iloc[:, 0]
    groups = temp.groupby(Grouper(freq=freq))

    boxes = []

    for group in groups:
        boxes.append(go.Box(
            name=str(group[0]),
            y=group[1]
        ))

    fig = go.Figure(data=boxes)
    fig.update_layout(title='Box plot', xaxis_title=df.index.name, yaxis_title='Count')

    g = dcc.Graph(
        figure=fig
    )
    return g


def prediction_plot(df: DataFrame, predicted_data: DataFrame, test_values: int) -> dcc.Graph:
    """
    Create and return a plot which contains the prediction for a dataframe.
    The plot is built using two dataframe: ingested_data and predicted_data.

    ingested_data includes the raw data ingested by the app, while predicted_data
    contains the actual prediction made by a model.

    Note that predicted_data starts at the first value used for training.

    The data not used for training is plotted in black, the data used for training
    is plotted in green and the test values are red.

    Note that predicted_data may or not have the columns "yhat_lower" and "yhat_upper".

    Parameters
    ----------
    df : DataFrame
    Raw values ingested by the app.

    predicted_data : DataFrame
    Prediction created by a model.

    test_values : int
    Number of test values used in the testing.

    Returns
    -------
    g : dcc.Graph
    """
    fig = go.Figure()

    not_training_data = df.loc[:predicted_data.index[0]]
    training_data = df.loc[predicted_data.index[0]:]
    training_data = training_data.iloc[:-test_values]
    test_data = df.iloc[-test_values:]

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
    fig.update_layout(title="Best prediction", xaxis_title=df.index.name,
                      yaxis_title=df.columns[0])
    g = dcc.Graph(
        figure=fig
    )
    return g


def performance_plot(df: DataFrame, predicted_data: DataFrame, testing_performances: [TestingPerformance],
                     test_values: int) -> dcc.Graph:
    """
    Create and return the performance plot of the model; for every error kind (i.e. MSE, MAE, etc)
    plot the values it assumes using different training windows.
    Plot the training data in the end.

    Parameters
    ----------
    df : DataFrame
    Raw values ingested by the app.

    predicted_data : DataFrame
    Prediction created by a model.

    testing_performances : [TestingPerformance]
    List of TestingPerformance object. Every object is related to a specific training windows, hence
    it shows the performance using that window.

    test_values : int
    Number of values used for testing performance.

    Returns
    -------
    g : dcc.Graph
    """
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    training_data = df.iloc[:-test_values]

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
    fig.update_yaxes(title_text=df.columns[0], row=3, col=1)

    fig.update_layout(title='Performances with different training windows', height=700)
    g = dcc.Graph(
        figure=fig
    )
    return g


def plot_every_prediction(df: DataFrame, model_results: [SingleResult],
                          main_accuracy_estimator: str, test_values: int):
    new_childrens = [html.Div("EXTRA: plot _EVERY_ prediction\n")]

    model_results.sort(key=lambda x: len(x.prediction))

    for r in model_results:
        predicted_data = r.prediction
        testing_performance = r.testing_performances
        plot = prediction_plot(df, predicted_data, test_values)
        plot.figure.update_layout(title="")
        new_childrens.extend([
            html.Div(main_accuracy_estimator.upper()
                     + ": " + str(getattr(testing_performance, main_accuracy_estimator.upper()))),
            plot
        ])

    return new_childrens


def characteristics_list(model_characteristics: dict, testing_performances: [TestingPerformance]) -> html.Div:
    """
    Create and return an HTML Div which contains a list of natural language characteristic
    relative to a prediction model.

    Parameters
    ----------
    model_characteristics : dict
    key-value for each characteristic to write in natural language.

    testing_performances : [TestingPerformance]
    Useful to write also information about the testing performances.

    Returns
    -------
    html.Div()
    """

    def get_text_char(key: str, value: any) -> str:
        switcher = {
            "name": "Model name: " + str(value),
            "test_values": "The last " + str(value) + " values have been used for testing.",
            "delta_training_percentage": "The length of the training windows is the " + str(value) + "% of the time "
                                                                                                     "series' length.",
            "delta_training_values": "Training windows are composed of " + str(value) + " values."
        }
        return switcher.get(key, "Invalid choice!")

    def get_text_perf(key: str, value: any) -> str:
        switcher = {
            "MAE": "MAE: " + str(round(value, 2)),
            "RMSE": "RMSE: " + str(round(value, 2)),
            "MSE": "MSE: " + str(round(value, 2)),
            "AM": "Arithmetic mean of errors: " + str(round(value, 2))
        }
        return switcher.get(key, "Invalid choice!")

    best_testing_performances = testing_performances[0].get_dict()
    del best_testing_performances["first_used_index"]

    elems = [html.Div("Model characteristics:"),
             html.Ul([html.Li(get_text_char(key, model_characteristics[key])) for key in model_characteristics]),
             html.Div("This model, using the best training window, reaches these performances:"),
             html.Ul(
                 [html.Li(get_text_perf(key, best_testing_performances[key])) for key in best_testing_performances])]

    return html.Div(elems)
