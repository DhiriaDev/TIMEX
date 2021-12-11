import logging
import gettext
import pathlib
import os

import pandas
from pandas import Grouper, DataFrame
import plotly.graph_objects as go
import numpy as np

import dash_core_components as dcc
import dash_html_components as html
from plotly.subplots import make_subplots
import networkx as nx
import dash_bootstrap_components as dbc

from colorhash import ColorHash
from statsmodels.tsa.seasonal import seasonal_decompose

from timexseries.data_prediction import ValidationPerformance
from timexseries.data_prediction.models.predictor import SingleResult
import calendar

from timexseries.timeseries_container import TimeSeriesContainer

log = logging.getLogger(__name__)

# Default method to get a translated text.
_ = lambda x: x


def create_timeseries_dash_children(timeseries_container: TimeSeriesContainer, param_config: dict):
    """
    Creates the Dash children for a specific time-series. They include a line plot, histogram, box plot and
    autocorrelation plot. For each model on the time-series the prediction plot and performance plot are also added.

    Cross-correlation plots and graphs are shown, if the the `timeseries_container` have it.

    If the `timeseries_container` also have the `historical_prediction`, it is shown in a plot.
    
    Parameters
    ----------
    timeseries_container: TimeSeriesContainer
        Time-series for which the various plots and graphs will be returned.
    
    param_config : dict
        TIMEX configuration parameters dictionary, used for `visualization_parameters` which contains settings to
        customize some plots and graphs.

    Returns
    -------
    list
        List of Dash children.

    Examples
    --------
    Given a `timexseries.timeseries_container.TimeSeriesContainer` object, obtained for example through
    `timexseries.data_prediction.pipeline.create_timeseries_containers`, create all the Dash object which could be shown in a
    Dash app:
    >>> param_config = {
    ...  "input_parameters": {},
    ...  "model_parameters": {
    ...      "models": "fbprophet",
    ...      "possible_transformations": "none,log_modified",
    ...      "main_accuracy_estimator": "mae",
    ...      "delta_training_percentage": 20,
    ...      "test_values": 5,
    ...      "prediction_lags": 7,
    ...  },
    ...  "historical_prediction_parameters": {
    ...      "initial_index": "2000-01-25",
    ...      "save_path": "example.pkl"
    ...  },
    ...  "visualization_parameters": {}
    ...}
    >>>  plots = create_timeseries_dash_children(timeseries_container, param_config)
    """
    children = []

    visualization_parameters = param_config["visualization_parameters"]
    timeseries_data = timeseries_container.timeseries_data

    name = timeseries_data.columns[0]

    locale_dir = pathlib.Path(os.path.abspath(__file__)).parent / "locales"

    global _
    try:
        gt = gettext.translation('messages', localedir=locale_dir, languages=[visualization_parameters["language"]])
        gt.install()
        _ = gt.gettext
    except:
        gt = gettext.translation('messages', localedir=locale_dir, languages=['en'])
        gt.install()
        _ = gt.gettext

    # Data visualization with plots
    children.extend([
        html.H2(children=name + _(' analysis'), id=name),
        html.H3(_("Data visualization")),
        line_plot(timeseries_data),
        histogram_plot(timeseries_data),
        box_plot(timeseries_data, visualization_parameters),
        box_plot_aggregate(timeseries_data, visualization_parameters),
        components_plot(timeseries_data),
        autocorrelation_plot(timeseries_data),
    ])

    # Plot cross-correlation plot and graphs, if requested.
    if timeseries_container.xcorr is not None:
        graph_corr_threshold = visualization_parameters[
            "xcorr_graph_threshold"] if "xcorr_graph_threshold" in visualization_parameters else None

        children.extend([
            html.H3(_("Cross-correlation")),
            html.Div(_("Negative lags (left part) show the correlation between this scenario and the future of the "
                       "others.")),
            html.Div(_("Meanwhile, positive lags (right part) shows the correlation between this scenario "
                       "and the past of the others.")),
            cross_correlation_plot(timeseries_container.xcorr),
            html.Div(_("The peaks found using each cross-correlation modality are shown in the graphs:")),
            cross_correlation_graph(name, timeseries_container.xcorr, graph_corr_threshold)
        ])

    # Plot the prediction results, if requested.
    if timeseries_container.models is not None:
        model_parameters = param_config["model_parameters"]

        models = timeseries_container.models

        children.append(
            html.H3(_("Training & Validation results")),
        )

        for model_name in models:
            model = models[model_name]
            model_results = model.results
            model_characteristic = model.characteristics

            test_values = model_characteristic["test_values"]
            main_accuracy_estimator = model_parameters["main_accuracy_estimator"]
            model_results.sort(key=lambda x: getattr(x.testing_performances, main_accuracy_estimator.upper()))

            best_prediction = model_results[0].prediction
            testing_performances = [x.testing_performances for x in model_results]

            children.extend([
                html.H4(f"{model_name}"),
                characteristics_list(model_characteristic, testing_performances),
                # html.Div("Testing performance:"),
                # html.Ul([html.Li(key + ": " + str(testing_performances[key])) for key in testing_performances]),
                prediction_plot(timeseries_data, best_prediction, test_values),
                performance_plot(timeseries_data, best_prediction, testing_performances, test_values),
            ])

            # EXTRA
            # Warning: this will plot every model result, with every training set used!
            # children.extend(plot_every_prediction(ingested_data, model_results, main_accuracy_estimator, test_values))

    if timeseries_container.historical_prediction is not None:
        children.extend([
            html.H3(_("Prediction")),
            html.Div(_("For every model the best predictions for each past date are plotted."))
        ])
        for model in timeseries_container.historical_prediction:
            children.extend([
                html.H4(f"{model}"),
                historical_prediction_plot(timeseries_data, timeseries_container.historical_prediction[model],
                                           timeseries_container.models[model].best_prediction)
            ])

    return children


def create_dash_children(timeseries_containers: [TimeSeriesContainer], param_config: dict):
    """
    Create Dash children, in order, for a list of `timexseries.timeseries_container.TimeSeriesContainer`.

    Parameters
    ----------
    timeseries_containers : [TimeSeriesContainer]
        Time-series for which all the plots and graphs will be created.

    param_config : dict
        TIMEX configuration parameters dictionary.

    Returns
    -------
    list
        List of Dash children.

    """
    children = []
    for s in timeseries_containers:
        children.extend(create_timeseries_dash_children(s, param_config))

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
        Dash object containing the line plot.

    Examples
    --------
    Get the `figure` attribute if you want to display this in a Jupyter notebook.
    >>> line_plot = line_plot(timeseries_container.timeseries_data).figure
    >>> line_plot.show()
    """
    fig = go.Figure(data=go.Scatter(x=df.index, y=df.iloc[:, 0], mode='lines+markers'))
    fig.update_layout(title=_('Line plot'), xaxis_title=df.index.name, yaxis_title=df.columns[0])

    g = dcc.Graph(
        figure=fig
    )
    return g


def line_plot_multiIndex(df: DataFrame) -> dcc.Graph:
    """
    Returns a line plot for a dataframe with a MultiIndex.
    It is assumed that the first-level index is the real index, and that data should be grouped using the
    second-level one.

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

    fig.update_layout(title=_('Line plot'), xaxis_title=df.index.get_level_values(0).name,
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

    Examples
    --------
    Get the `figure` attribute if you want to display this in a Jupyter notebook.
    >>> hist_plot = hist_plot(timeseries_container.timeseries_data).figure
    >>> hist_plot.show()
    """

    max_value = abs(df.iloc[:, 0].max() - df.iloc[:, 0].min())
    step = max_value / 400

    fig = go.Figure(data=[go.Histogram(x=df.iloc[:, 0])])
    fig.layout.sliders = [dict(
        steps=[dict(method='restyle', args=['xbins.size', i]) for i in np.arange(step, max_value / 2, step)],
        font=dict(color="rgba(0,0,0,0)"),
        tickcolor="rgba(0,0,0,0)"
    )]

    fig.update_layout(title=_('Histogram'), xaxis_title_text=df.columns[0], yaxis_title_text=_('Count'))
    g = dcc.Graph(
        figure=fig
    )
    return g


def components_plot(ingested_data: DataFrame) -> html.Div:
    """
    Create and return the plots of all the components of the time series: level, trend, residual.
    It uses both an additive and multiplicative model, with a subplot.

    Parameters
    ----------
    ingested_data : DataFrame
        Original time series values.

    Returns
    -------
    g : dcc.Graph

    Examples
    --------
    Get the `figure` attribute if you want to display this in a Jupyter notebook.
    >>> comp_plot = components_plot(timeseries_container.timeseries_data)[0].figure
    >>> comp_plot.show()
    """
    modes = ["additive", "multiplicative"]

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[_("Trend"), _("Seasonality"), _("Residual")], shared_xaxes=True, vertical_spacing=0.05,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]]
    )

    interpolated = ingested_data.interpolate()
    interpolated = interpolated.fillna(0)

    for mode in modes:
        try:
            result = seasonal_decompose(interpolated, model=mode)
            trend = result.trend
            seasonal = result.seasonal
            residual = result.resid

            secondary_y = False if mode == "additive" else True

            fig.add_trace(go.Scatter(x=trend.index, y=trend,
                                     mode='lines+markers',
                                     name=_(mode.capitalize()), legendgroup=_(mode.capitalize()),
                                     line=dict(color=ColorHash(mode).hex)),
                          row=1, col=1, secondary_y=secondary_y)
            fig.add_trace(go.Scatter(x=seasonal.index, y=seasonal,
                                     mode='lines+markers', showlegend=False,
                                     name=_(mode.capitalize()), legendgroup=_(mode.capitalize()),
                                     line=dict(color=ColorHash(mode).hex)),
                          row=2, col=1, secondary_y=secondary_y)
            fig.add_trace(go.Scatter(x=residual.index, y=residual,
                                     mode='lines+markers', showlegend=False,
                                     name=_(mode.capitalize()), legendgroup=_(mode.capitalize()),
                                     line=dict(color=ColorHash(mode).hex)),
                          row=3, col=1, secondary_y=secondary_y)
        except ValueError:
            log.warning(f"Multiplicative decomposition not available for {ingested_data.columns[0]}")

    fig.update_layout(title=_("Components decomposition"), height=1000, legend_title_text=_('Decomposition model'))
    fig.update_yaxes(title_text="<b>" + _('Additive') + "</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>" + _('Multiplicative') + "</b>", secondary_y=True)

    g = dcc.Graph(
        figure=fig
    )

    warning = html.H5(_("Multiplicative model is not available for series which contain zero or negative values."))

    return html.Div([g, warning])


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

    Examples
    --------
    Get the `figure` attribute if you want to display this in a Jupyter notebook.
    >>> auto_plot = autocorrelation_plot(timeseries_container.timeseries_data).figure
    >>> auto_plot.show()
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
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=_('autocorrelation')))
    fig.add_trace(go.Scatter(x=x, y=np.full(n, c1), line=dict(color='gray', width=1), name='z99'))
    fig.add_trace(go.Scatter(x=x, y=np.full(n, c2), line=dict(color='gray', width=1), name='z95'))
    fig.add_trace(go.Scatter(x=x, y=np.full(n, c3), line=dict(color='gray', width=1), name='-z95'))
    fig.add_trace(go.Scatter(x=x, y=np.full(n, c4), line=dict(color='gray', width=1), name='-z99'))
    fig.update_layout(title=_('Autocorrelation plot'), xaxis_title=_('Lags'), yaxis_title=_('Autocorrelation'))
    fig.update_yaxes(tick0=-1.0, dtick=0.25)
    fig.update_yaxes(range=[-1.2, 1.2])
    g = dcc.Graph(
        figure=fig
    )
    return g


def cross_correlation_plot(xcorr: dict):
    """
    Create and return the cross-correlation plot for all the columns in the dataframe.
    The time-series column is used as target; the correlation is shown in a subplot for every modality used to compute
    the x-correlation.

    Parameters
    ----------
    xcorr : dict
        Cross-correlation values.

    Returns
    -------
    g : dcc.Graph

    Examples
    --------
    Get the `figure` attribute if you want to display this in a Jupyter notebook.
    >>> xcorr_plot = cross_correlation_plot(timeseries_container.xcorr).figure
    >>> xcorr_plot.show()
    """
    subplots = len(xcorr)
    combs = [(1, 1), (1, 2), (2, 1), (2, 2)]

    rows = 1 if subplots < 3 else 2
    cols = 1 if subplots < 2 else 2

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=([*xcorr.keys()]))

    i = 0
    for mode in xcorr:
        for col in xcorr[mode].columns:
            fig.add_trace(go.Scatter(x=xcorr[mode].index, y=xcorr[mode][col],
                                     mode='lines',
                                     name=col, legendgroup=col, line=dict(color=ColorHash(col).hex),
                                     showlegend=True if i == 0 else False),
                          row=combs[i][0], col=combs[i][1])
        i += 1

    # Formula from https://support.minitab.com/en-us/minitab/18/help-and-how-to/modeling-statistics/time-series/how-to/cross-correlation/interpret-the-results/all-statistics-and-graphs/
    # significance_level = DataFrame(columns=['Value'], dtype=np.float64)
    # for i in range(-lags, lags):
    #     significance_level.loc[i] = 2 / np.sqrt(lags - abs(i))

    # fig.add_trace(
    #     go.Scatter(x=significance_level.index, y=significance_level['Value'], line=dict(color='gray', width=1), name='z95'))
    # fig.add_trace(
    #     go.Scatter(x=significance_level.index, y=-significance_level['Value'], line=dict(color='gray', width=1), name='-z95'))

    fig.update_layout(title=_("Cross-correlation using different algorithms"))
    fig.update_xaxes(title_text=_("Lags"))
    fig.update_yaxes(tick0=-1.0, dtick=0.25, range=[-1.2, 1.2], title_text=_("Correlation"))

    g = dcc.Graph(
        figure=fig
    )
    return g


def cross_correlation_graph(name: str, xcorr: dict, threshold: float = 0) -> dcc.Graph:
    """
    Create and return the cross-correlation graphs for all the columns in the dataframe.
    A graph is created for each mode used to compute the x-correlation.

    The nodes are all the time-series which can be found in `xcorr`; an arc is drawn from `target` node to another node
    if the cross-correlation with that time-series, at any lag, is above the `threshold`. The arc contains also the
    information on the lag.

    Parameters
    ----------
    name : str
        Name of the target.

    xcorr : dict
        Cross-correlation dataframe.

    threshold : int
        Minimum value of correlation for which a edge should be drawn. Default 0.

    Returns
    -------
    g : dcc.Graph

    Examples
    --------
    This is thought to be shown in a Dash app, so it could be difficult to show in Jupyter.
    >>> xcorr_graph = cross_correlation_graph("a", timeseries_container.xcorr, 0.7)
    """
    figures = []

    i = 0
    for mode in xcorr:
        G = nx.DiGraph()
        G.add_nodes_from(xcorr[mode].columns)
        G.add_node(name)

        for col in xcorr[mode].columns:
            index_of_max = xcorr[mode][col].abs().idxmax()
            corr = xcorr[mode].loc[index_of_max, col]
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

        graph = go.Figure(data=[node_trace, edge_trace],
                          layout=go.Layout(title=str(mode),
                                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                           showlegend=False,
                                           annotations=annotateArrows,
                                           height=400, margin=dict(l=10, r=10, t=50, b=30)))

        # Add annotations on edges
        for e in G.edges:
            lag = str(G.edges[e]['lag'])
            corr = str(round(G.edges[e]['corr'], 3))

            end = e[1]
            x, y = pos.get(end)

            graph.add_annotation(x=x, y=y, text=_("Lag: ") + lag + ", corr: " + corr, yshift=20, showarrow=False,
                                 bgcolor='white')

        figures.append(graph)
        i += 1

    n_graphs = len(figures)
    if n_graphs == 1:
        g = dcc.Graph(figure=figures[0])
    elif n_graphs == 2:
        g = html.Div(dbc.Row([
            dbc.Col(dcc.Graph(figure=figures[0])),
            dbc.Col(dcc.Graph(figure=figures[1]))
        ]))
    elif n_graphs == 3:
        g = html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=figures[0])),
                dbc.Col(dcc.Graph(figure=figures[1]))
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=figures[2]))
            ])
        ])
    elif n_graphs == 4:
        g = html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=figures[0])),
                dbc.Col(dcc.Graph(figure=figures[1])),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=figures[2])),
                dbc.Col(dcc.Graph(figure=figures[3]))
            ])
        ])
    else:
        g = html.Div()

    return g


def box_plot(df: DataFrame, visualization_parameters: dict) -> dcc.Graph:
    """
    Create and return the box-and-whisker plot for a dataframe.

    Parameters
    ----------
    df : DataFrame
        Dataframe to use in the box plot.

    visualization_parameters : dict
        Options set by the user. In particular, `box_plot_frequency` is used to determine the number of boxes. The
        default value is `1W`.

    Returns
    -------
    g : dcc.Graph

    Examples
    --------
    Get the `figure` attribute if you want to display this in a Jupyter notebook.
    >>> box_plot = box_plot(timeseries_container.timeseries_data, param_config["visualization_parameters"]).figure
    >>> box_plot.show()
    """
    temp = df.iloc[:, 0]

    aggregations = ["1W", "2W", "1M", "3M", "4M", "6M", "1Y"]

    try:
        initial_freq = visualization_parameters['box_plot_frequency']
    except KeyError:
        initial_freq = '1W'

    traces = []

    for freq in aggregations:
        is_visible = True if initial_freq == freq else 'legendonly'

        # Small trick needed to show `name` in legend.
        traces.append(go.Scatter(x=[None], y=[None], legendgroup=freq, name=freq,
                                 visible=is_visible))

    for freq in aggregations:
        groups = temp.groupby(Grouper(freq=freq))
        is_visible = True if initial_freq == freq else 'legendonly'

        for group in groups:
            traces.append(go.Box(
                name=str(group[0].normalize()),
                y=group[1], legendgroup=freq, showlegend=False, visible=is_visible, xperiodalignment='end'
            ))

    fig = go.Figure(traces)
    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            direction='right',
            yanchor='bottom',
            xanchor='left',
            y=1,
            active=aggregations.index(_(initial_freq)),
            buttons=list(
                [dict(label=f, method='update', args=[{'visible': [tr.legendgroup == f for tr in traces]}])
                 for f in aggregations]))])

    fig.update_layout(title=_('Box plot'), xaxis_title=df.index.name, yaxis_title=_('Count'), showlegend=False)
    fig.update_xaxes(type='category')
    fig.add_annotation(text=_("The index refer to the end of the period."),
                       xref="paper", yref="paper",
                       x=1.0, y=1.0, yanchor='bottom', showarrow=False)

    g = dcc.Graph(
        figure=fig
    )
    return g


def box_plot_aggregate(df: DataFrame, visualization_parameters: dict) -> dcc.Graph:
    """
    Create and return the aggregate box plot for a dataframe, i.e. a box plot which shows, for each day of the week/for
    each month of the year the distribution of the values. Now also with aggregation over minute, hour, week, month,
    year.

    Parameters
    ----------
    df : DataFrame
        Dataframe to use in the box plot.

    visualization_parameters : dict
        Options set by the user. In particular, `aggregate_box_plot_frequency` is used. Default `weekday`. It controls
        which frequency is activated at launch.

    Returns
    -------
    g : dcc.Graph

    Examples
    --------
    Get the `figure` attribute if you want to display this in a Jupyter notebook.
    >>> abox_plot = box_plot_aggregate(timeseries_container.timeseries_data,
    ...                                param_config["visualization_parameters"]).figure
    >>> abox_plot.show()
    """
    aggregations = ["minute", "hour", "weekday", "day", "month", "year"]
    translated_aggregations = [_("minute"), _("hour"), _("weekday"), _("day"), _("month"), _("year")]

    temp = df.iloc[:, 0]

    try:
        initial_freq = visualization_parameters['aggregate_box_plot_frequency']
    except KeyError:
        initial_freq = 'weekday'

    traces = []

    for freq, translated_freq in zip(aggregations, translated_aggregations):
        is_visible = True if initial_freq == freq else False

        # Small trick needed to show `name` in legend.
        traces.append(go.Scatter(x=[None], y=[None], legendgroup=translated_freq, name=translated_freq,
                                 visible=is_visible))

    for freq, translated_freq in zip(aggregations, translated_aggregations):
        groups = temp.groupby(getattr(temp.index, freq))
        is_visible = True if initial_freq == freq else False

        if freq == "weekday":
            for group in groups:
                traces.append(go.Box(
                    name=calendar.day_name[group[0]],
                    y=group[1], legendgroup=translated_freq, showlegend=False, visible=is_visible
                ))
        elif freq == "month":
            for group in groups:
                traces.append(go.Box(
                    name=calendar.month_name[group[0]],
                    y=group[1], legendgroup=translated_freq, showlegend=False, visible=is_visible
                ))
        else:
            for group in groups:
                traces.append(go.Box(
                    name=group[0],
                    y=group[1], legendgroup=translated_freq, showlegend=False, visible=is_visible
                ))

    fig = go.Figure(data=traces)

    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            direction='right',
            yanchor='bottom',
            xanchor='left',
            y=1,
            active=translated_aggregations.index(_(initial_freq)),
            buttons=list(
                [dict(label=f, method='update', args=[{'visible': [tr.legendgroup == f for tr in traces]}])
                 for f in translated_aggregations]))])

    fig.update_layout(title=_('Aggregate box plot'), yaxis_title=_('Count'), showlegend=False)
    fig.update_xaxes(type='category')

    g = dcc.Graph(
        figure=fig
    )
    return g


def prediction_plot(df: DataFrame, predicted_data: DataFrame, test_values: int = 0) -> dcc.Graph:
    """
    Create and return a plot which contains the prediction for a dataframe.
    The plot is built using two dataframe: `ingested_data` and `predicted_data`.

    `ingested_data` includes the raw data ingested by the app, while `predicted_data` contains the actual prediction
    made by a model.

    Note that `predicted_data` starts at the first value used for training.

    The data not used for training is plotted in black, the data used for training is plotted in green and the
    validation values dashed.

    Note that `predicted_data` may or not have the columns "yhat_lower" and "yhat_upper".

    Parameters
    ----------
    df : DataFrame
        Raw values ingested by the app.

    predicted_data : DataFrame
        Prediction created by a model.

    test_values : int, optional, default 0
        Number of validation values used in the testing.

    Returns
    -------
    g : dcc.Graph

    See Also
    --------
    Check `create_timeseries_dash_children` to check the use.
    """
    fig = go.Figure()

    not_training_data = df.loc[:predicted_data.index[0]]
    training_data = df.loc[predicted_data.index[0]:]
    training_data = training_data.iloc[:-test_values]

    fig.add_trace(go.Scatter(x=not_training_data.index, y=not_training_data.iloc[:, 0],
                             line=dict(color='black'),
                             mode='markers',
                             name=_('unused data')))
    fig.add_trace(go.Scatter(x=training_data.index, y=training_data.iloc[:, 0],
                             line=dict(color='green', width=4, dash='dash'),
                             mode='markers',
                             name=_('training data'),
                             ))

    if test_values > 0:
        test_data = df.iloc[-test_values:]
        fig.add_trace(go.Scatter(x=test_data.index, y=test_data.iloc[:, 0],
                                 line=dict(color='green', width=3, dash='dot'),
                                 name=_('validation data')))

    fig.add_trace(go.Scatter(x=predicted_data.index, y=predicted_data['yhat'],
                             line=dict(color='blue'),
                             mode='lines+markers',
                             name=_('yhat')))
    try:
        fig.add_trace(go.Scatter(x=predicted_data.index, y=predicted_data['yhat_lower'],
                                 line=dict(color='lightgreen', dash='dash'),
                                 name=_('yhatlower')))
        fig.add_trace(go.Scatter(x=predicted_data.index, y=predicted_data['yhat_upper'],
                                 line=dict(color='lightgreen', dash='dash'),
                                 name=_('yhatupper')))
    except:
        pass

    fig.update_layout(title=_("Best prediction for the validation set"), xaxis_title=df.index.name,
                      yaxis_title=df.columns[0])
    g = dcc.Graph(
        figure=fig
    )
    return g


def historical_prediction_plot(real_data: DataFrame, historical_prediction: DataFrame,
                               future_prediction: DataFrame) -> html.Div:
    """
    Create and return a plot which contains the best prediction found by this model for this time series, along with
    the historical prediction. The plot of the error is also drawn.

    Note that `predicted_data` may or not have the columns "yhat_lower" and "yhat_upper".

    Parameters
    ----------
    real_data : DataFrame
        Raw values ingested by the app.

    historical_prediction : DataFrame
        Historical prediction.

    future_prediction : DataFrame
        Best prediction, corresponding to the `best_prediction` attribute of a
        `timexseries.data_prediction.models.predictor.ModelResult`.

    Returns
    -------
    g : dcc.Graph

    See Also
    --------
    Check `create_timeseries_dash_children` to check the use.
    """
    new_children = []
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    timeseries_name = real_data.columns[0]
    first_predicted_index = historical_prediction.index[0]
    last_real_index = real_data.index[-1]

    validation_real_data = real_data.loc[first_predicted_index:, timeseries_name]

    testing_performance = ValidationPerformance(first_predicted_index)
    testing_performance.set_testing_stats(actual=validation_real_data,
                                          predicted=historical_prediction.loc[:last_real_index, timeseries_name])
    new_children.extend([
        html.Div(_("This model, during the history, reached these performances on unseen data:")),
        show_errors(testing_performance),
        html.Div(_("Range of validation data: ") + str(validation_real_data.max() - validation_real_data.min()))])

    fig.add_trace(go.Scatter(x=real_data.index, y=real_data.iloc[:, 0],
                             line=dict(color='red'),
                             mode='markers',
                             name=_('real data')), row=1, col=1)

    fig.add_trace(go.Scatter(x=historical_prediction.index, y=historical_prediction.iloc[:, 0],
                             line=dict(color='blue'),
                             mode='lines+markers',
                             name=_('historical prediction')), row=1, col=1)

    future_prediction.loc[historical_prediction.index[-1], 'yhat'] = historical_prediction.iloc[-1, 0]
    future_prediction = future_prediction.loc[historical_prediction.index[-1]:, :]

    fig.add_trace(go.Scatter(x=future_prediction.index, y=future_prediction['yhat'],
                             line=dict(color='lightgreen'),
                             mode='lines+markers',
                             name=_('future yhat')), row=1, col=1)

    error_series = validation_real_data - historical_prediction.iloc[:, 0]
    fig.add_trace(go.Scatter(x=error_series.index, y=error_series,
                             line=dict(color='black'),
                             mode='lines+markers',
                             name=_("Error series")), row=2, col=1)

    fig.update_yaxes(title_text=_("Historical prediction"), row=1, col=1)
    fig.update_yaxes(title_text=_("Error series"), row=2, col=1)

    fig.update_xaxes(title_text=real_data.index.name, row=2, col=1)
    fig.update_layout(title=_("Historical prediction"), height=900)

    g = dcc.Graph(
        figure=fig
    )
    new_children.append(g)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=error_series, histnorm="probability",
                                    marker=dict(color='black'), name=_("Error series histogram")))
    error_max_value = abs(error_series.max() - error_series.min())
    step = error_max_value / 200
    fig_hist.layout.sliders = [dict(
        steps=[dict(method='restyle', args=['xbins.size', i]) for i in np.arange(step, error_max_value / 2, step)],
        font=dict(color="rgba(0,0,0,0)"),
        tickcolor="rgba(0,0,0,0)"
    )]
    fig_hist.update_yaxes(title_text=_("Error series histogram"), tickformat='.1%')
    fig_hist.update_xaxes(title_text=timeseries_name)

    g = dcc.Graph(
        figure=fig_hist
    )
    new_children.append(g)

    return html.Div(new_children)


def performance_plot(df: DataFrame, predicted_data: DataFrame, testing_performances: [ValidationPerformance],
                     test_values: int) -> dcc.Graph:
    """
    Create and return the performance plot of the model; for every error kind (i.e. MSE, MAE, etc) plot the values it
    assumes using different training windows.
    Plot the training data in the end.

    Parameters
    ----------
    df : DataFrame
        Raw values ingested by the app.

    predicted_data : DataFrame
        Prediction created by a model.

    testing_performances : [ValidationPerformance]
        List of ValidationPerformance object. Every object is related to a specific training windows, hence
        it shows the performance using that window.

    test_values : int
        Number of values used for testing performance.

    Returns
    -------
    g : dcc.Graph

    See Also
    --------
    Check `create_timeseries_dash_children` to check the use.
    """
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    training_data = df.iloc[:-test_values]

    data_performances = []

    for tp in testing_performances:
        data_performances.append([tp.first_used_index, tp.MAE, tp.MSE, tp.AM])

    df_performances = pandas.DataFrame(data_performances, columns=['index', 'mae', 'mse', 'am'])
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

    fig.append_trace(go.Scatter(x=df_performances.index, y=df_performances['am'],
                                line=dict(color='blue'),
                                mode="lines+markers",
                                name='AM'), row=3, col=1)

    fig.append_trace(go.Scatter(x=training_data.index, y=training_data.iloc[:, 0],
                                line=dict(color='black'),
                                mode='markers',
                                name=_('training data')), row=4, col=1)

    # Small trick to make the x-axis have the same length of the "Prediction plot"
    predicted_data.iloc[:, 0] = "nan"
    fig.append_trace(go.Scatter(x=predicted_data.index, y=predicted_data.iloc[:, 0],
                                mode='lines+markers',
                                name='yhat', showlegend=False), row=4, col=1)

    fig.update_yaxes(title_text="MAE", row=1, col=1)
    fig.update_yaxes(title_text="MSE", row=2, col=1)
    fig.update_yaxes(title_text="AM", row=3, col=1)
    fig.update_yaxes(title_text=df.columns[0], row=4, col=1)

    fig.update_layout(title=_('Performances with different training windows'), height=900)
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


def characteristics_list(model_characteristics: dict, testing_performances: [ValidationPerformance]) -> html.Div:
    """
    Create and return an HTML Div which contains a list of natural language characteristic
    relative to a prediction model.

    Parameters
    ----------
    model_characteristics : dict
        key-value for each characteristic to write in natural language.

    testing_performances : [ValidationPerformance]
        Useful to write also information about the testing performances.

    Returns
    -------
    html.Div()
    """

    def get_text_char(key: str, value: any) -> str:
        value = str(value)
        switcher = {
            "name": _("Model type: ") + value,
            "test_values": _('Values used for testing: last ') + value + _(' values'),
            "delta_training_percentage": _('The length of the training windows is the ') + value
                                         + "%" + _(' of the length of the time series.'),
            "delta_training_values": _('Training windows are composed of ') + value + _(' values.'),
            "extra_regressors": _("The model has used ") + value + _(" as extra-regressor(s) to improve the training."),
            "transformation": _('The model has used a ') + value + _(
                ' transformation on the input data.') if value != "none "
            else _('The model has not used any pre/post transformation on input data.')
        }
        return switcher.get(key, "Invalid choice!")

    elems = [html.Div(_('Model characteristics:')),
             html.Ul([html.Li(get_text_char(key, model_characteristics[key])) for key in model_characteristics]),
             html.Div(_("This model, using the best training window, reaches these performances:")),
             show_errors(testing_performances[0])]

    return html.Div(elems)


def show_errors(testing_performances: ValidationPerformance) -> html.Ul:
    """
    Create an HTML list with each error-metric in `testing_performances`.

    Parameters
    ----------
    testing_performances : ValidationPerformance
        Error metrics to show.

    Returns
    -------
    html.Ul
        HTML list with all the error-metrics.
    """
    import math

    def round_n(n: float):
        dec_part, int_part = math.modf(n)

        if abs(int_part) > 1:
            return str(round(n, 3))
        else:
            return format(n, '.3g')

    def get_text_perf(key: str, value: any) -> str:
        switcher = {
            "MAE": "MAE: " + round_n(value),
            "RMSE": "RMSE: " + round_n(value),
            "MSE": "MSE: " + round_n(value),
            "AM": _('Arithmetic mean of errors:') + round_n(value),
            "SD": _('Standard deviation of errors: ') + round_n(value)
        }
        return switcher.get(key, "Invalid choice!")

    testing_performances = testing_performances.get_dict()
    del testing_performances["first_used_index"]

    return html.Ul([html.Li(get_text_perf(key, testing_performances[key])) for key in testing_performances])
