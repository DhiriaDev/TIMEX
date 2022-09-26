import pickle
from datetime import datetime, timezone

import dash
from dash import html
from datetime import datetime, timezone
from dash import dcc
import dateparser
from dash.dependencies import Input, Output
from pandas import read_csv

from timexseries.data_ingestion import ingest_timeseries
from timexseries.data_prediction import create_timeseries_containers
from timexseries.data_visualization import create_timeseries_dash_children

param_config = {
        "activity_title": "Example",
        "verbose": "INFO",
        "input_parameters": {
            "source_data_url": "test1.csv",
            "datetime_column_name": "ind",
            "index_column_name": "ind",
            "frequency": "D",
        },
        "model_parameters": {
            "test_values": 3,
            "delta_training_percentage": 30,
            "prediction_lags": 5,
            "possible_transformations": "none,log_modified",
            "models": "mockup",
            "main_accuracy_estimator": "mae"
        },
        "historical_prediction_parameters": {
            "initial_index": "2000-01-15",
            "save_path": f"historical_predictions.pkl"
        },
        "visualization_parameters": {
            "xcorr_graph_threshold": 0.8,
            "box_plot_frequency": "1W"
        }
    }

ingested_dataset = ingest_timeseries(param_config)
timeseries_containers = create_timeseries_containers(ingested_dataset, param_config)

# Data visualization
children_for_each_timeseries = [{
    'name': s.timeseries_data.columns[0],
    'children': create_timeseries_dash_children(s, param_config)
} for s in timeseries_containers]

with open("expected_children1.pkl", 'wb') as f:
    pickle.dump(children_for_each_timeseries, f)

# Initialize Dash app.
app = dash.Dash(__name__)
server = app.server

now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

disclaimer = [html.Div([
    html.H1("Test"),
    html.Hr(),
    dcc.Markdown('''
        Test
        '''),
    html.Div("Last updated at (yyyy-mm-dd, UTC time): " + str(now)),
    html.Br(),
    html.H2("Please select the data of interest:")
], style={'width': '80%', 'margin': 'auto'}
), dcc.Dropdown(
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


app.run_server()
