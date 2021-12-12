from app import app

import os
import logging
import sys

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import json
import pandas as pd
import base64
import io

from timexseries.data_ingestion import ingest_timeseries
from timexseries.data_prediction.pipeline import create_timeseries_containers
from timexseries.data_visualization.functions import create_timeseries_dash_children


# ---------------------GLOBAL VARIABLES AND SOME USEFUL FUNCTIONS-----------------------------
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)
uploadStyle = {
    'width': '20%',
    'height': '60px',
    'lineHeight': '60px',
    'borderWidth': '1px',
    'borderStyle': 'dashed',
    'borderRadius': '5px',
    'textAlign': 'center',
    'margin-top' : 5
}
# this variable must be defined here to be shared by all the callbacks
children_for_each_timeseries = None


def prepare_folder_for_dataSaving(path: str):
    if(not os.path.exists(path)):
        cur = ""
        print('now i will try find the path')
        for dir in (path.split('/')[:-1]):
            cur += dir
            if(not os.path.exists(cur)):
                print('not present')
                os.makedirs(cur)


# -------------------------PAGE START----------------------------------------------------
app.layout = html.Div([
    html.Div([
        html.H1('Welcome to Timex - Time Series Forecasting a.a.S.'),
        html.Br(),

        html.Div(
            children=[
                html.H3( html.Label(children='Please insert here your configuration json',
                           htmlFor='uploaded')),
                dcc.Upload(html.A('Select a File'), id='uploaded',
                           style=uploadStyle, multiple=False),
            ]
        ),

        html.Hr(),
        html.H3(id='error_sentence', style={"color": "red"}),
    ], id='Welcome Div'),

    html.Div(id='outputDiv')

], style={'width':'80%', 'padding': 50})

# -------------------------FIRST CALLBACK AFTER THE INSERTION OF THE INPUT FOR THE PREDICTION PHASE------------------------


@ app.callback(
    Output(component_id='error_sentence', component_property='children'),
    Output(component_id='outputDiv', component_property='children'),
    Input(component_id='uploaded', component_property='contents'),
    State(component_id='uploaded', component_property='filename'),
    # it prevents the callback from being executed when the page is refreshed
    prevent_initial_call=True
)
def configuration_ingestion(config_file, filename):

    if(config_file is not None and filename is not None):

        encoding = 'utf-8'

        if 'json' in filename:
            try:
                content_type, content_string = config_file.split(',')

                json_config = json.load(
                    io.StringIO(((base64.b64decode(content_string)).decode(encoding))))

                original_url = (json_config['input_parameters'])[
                    'source_data_url']
                (json_config['input_parameters'])[
                    'source_data_url'] = 'https://drive.google.com/uc?id=' + original_url.split('/')[-2] #parsing the google drive link
            except ValueError as e:
                print(e)
                return 'Error in parsing the json file', dash.no_update
        else:
            return 'Wrong file format, please retry!', dash.no_update

        if(json_config is None):
            return 'Updated failed, please retry', dash.no_update
        else:
            return dash.no_update, compute_prediction(json_config)
    else:
        return 'Updated failed, please retry', dash.no_update

# This function computes the prediction and creates the children of the output div


def compute_prediction(param_config: dict):
    print('compute predictions...')
    prepare_folder_for_dataSaving(param_config['historical_prediction_parameters']['save_path'])

    ingested_dataset = ingest_timeseries(param_config)

    timeseries_containers = create_timeseries_containers(
        ingested_dataset, param_config)

    global children_for_each_timeseries

    children_for_each_timeseries = [{
        'name': s.timeseries_data.columns[0],
        'children': create_timeseries_dash_children(s, param_config)
    } for s in timeseries_containers]

    outputDiv_children = [
        html.H2(param_config["activity_title"]),
        html.H2("Please select the data of interest:"),
        dcc.Dropdown(
            id='timeseries_selector',
            options=[{'label': i['name'], 'value': i['name']}
                     for i in children_for_each_timeseries],
            value='Time-series'
        ), html.Div(id="timeseries_wrapper"), html.Div(dcc.Graph(), style={'display': 'none'})
    ]

    return outputDiv_children


# -------------------------SECOND CALLBACK FOR AN INTERACTIVE DISPLAYING OF THE RESULTS-------------------------
@ app.callback(
    Output(component_id='timeseries_wrapper', component_property='children'),
    [Input(component_id='timeseries_selector', component_property='value')],

    prevent_initial_call=True
)
def update_timeseries_wrapper(input_value):
    global children_for_each_timeseries

    try:
        children = next(
            x['children'] for x in children_for_each_timeseries if x['name'] == input_value)
    except StopIteration:
        return html.Div(style={'padding': 200})

    return children


if __name__ == '__main__':
    app.run_server(debug=True)
