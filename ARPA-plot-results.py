
param_config = {
    "activity_title": "ARPA pollution",
    "verbose": "INFO",
    "service" : "timex",
    "input_parameters": {
        "source_data_url": "../TimexDocker/dataset_examples/ARPA/ARPA-Milano.csv",
        "columns_to_load_from_url": "Date,PM10(µg/m³),PM2.5(µg/m³),nitrogen-dioxide(µg/m³),Ozone(µg/m³),BlackCarbon(µg/m³)",
        "datetime_column_name": "Date",
        "index_column_name": "Date",
        "frequency": "D"
    },
    "model_parameters": {
        'validation_values':15,   # Use the last 15 real values of the time-series as test data, to check the performances.
        "delta_training_percentage": 30,  # Training windows are composed of the 15% of the time-series length; more about this later...
        "forecast_horizon": 15,  # Predict the next 10 days.
        "possible_transformations" : "none,log_modified",  # Try to use no transformation or a logarithmic one.
        "models": "fbprophet,seasonal_persistence",  # Use models of class Facebook Prophet.
        "main_accuracy_estimator": "rmse"  # Use the Mean Absolute Error as main metric to measure accuracy.
    },
     "xcorr_parameters": {
        "xcorr_max_lags": 10,
        "xcorr_extra_regressor_threshold": 0.0,
        "xcorr_mode": "pearson",
        "xcorr_mode_target": "pearson"
    },
    "visualization_parameters": {
        "xcorr_graph_threshold": 0.0,
        "box_plot_frequency": "1W"
    }
  
}


import pickle
with open('ARPA-timeseriesContainerProphetSeasPers.pkl', 'rb') as f:
    tsc = pickle.load(f)


    
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from timexseries.data_visualization.functions import create_timeseries_dash_children
from jupyter_dash import JupyterDash

import dash_core_components as dcc
from dash.dependencies import Input, Output

# Data visualization
children_for_each_timeseries = [{
    'name': s.timeseries_data.columns[0],
    'children': create_timeseries_dash_children(s, param_config)
} for s in tsc]  # This creates the Dash children, for each time-series in the dataset (in this case just one!)

# Initialize Dash app.
app = JupyterDash(__name__)
# app = dash.Dash(__name__) Use this outside of Jupyter!

server = app.server

disclaimer = [html.Div([
    html.H1("Bitcoin price: monitoring and forecasting", style={'text-align': 'center'}),
    html.Hr(),
    html.H4(
        "Dashboard by the Intelligent Embedded Systems (IES) research group of the Politecnico di Milano, Italy",
        style={'text-align': 'center', 'top-margin': '25px'}),
    html.Hr(),
    dcc.Markdown('''
        *Welcome to the monitoring and forecasting dashboard of the Bitcoin USD/BTC price!*
        '''),
    html.Br(),
    html.H2("Please select the data of interest:")
    ], style={'width': '80%', 'margin': 'auto'}
    ), 
dcc.Dropdown(
    id='timeseries_selector',
    options=[{'label': i['name'], 'value': i['name']} for i in children_for_each_timeseries],
    value='Time-series'
), html.Div(id="timeseries_wrapper"), html.Div(dcc.Graph(), style={'display': 'none'})]
tree = html.Div(children=disclaimer, style={'width': '80%', 'margin': 'auto'})

app.layout = tree


@app.callback(
    Output(component_id='timeseries_wrapper', component_property='children'),
    [Input(component_id='timeseries_selector', component_property='value')]
)
def update_timeseries_wrapper(input_value):
    try:
        children = next(x['children'] for x in children_for_each_timeseries if x['name'] == input_value)
    except StopIteration:
        return html.Div(style={'padding': 200})

    return children

app.run_server(port=10000, debug=True)
