from datetime import time
import logging
import sys

import flask
from flask import Flask, request
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import json
import pickle
import base64
import requests
import io
from data_visualization import timeseries_container


from data_visualization.functions import create_timeseries_dash_children


# -----------SERVER INIT-----------------------

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

server = Flask(__name__)

app = dash.Dash(__name__,
                server=server,
                suppress_callback_exceptions=True,
                external_stylesheets=external_stylesheets,
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )



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
    'margin-top': 5
}
# these variables must be defined here to be shared by all the callbacks
children_for_each_timeseries = None

app.layout = html.Div([
    html.Div([
        html.H1('Welcome to Timex - Time Series Forecasting a.a.S.'),
        html.Br(),

        html.Div(
            children=[
                html.H3(html.Label(children='Please insert here your configuration json',
                                    htmlFor='uploaded')),
                dcc.Upload(html.A('Select a File'), id='uploaded',
                            style=uploadStyle, multiple=False),
            ]
        ),

        html.Hr(),
        html.H3(id='error_sentence', style={"color": "red"}),
    ], id='Welcome Div'),

    html.Div(id='outputDiv')

], style={'width': '80%', 'padding': 50})


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

                param_config = json.load(
                    io.StringIO(((base64.b64decode(content_string)).decode(encoding))))

                global prediction_parameters
                prediction_parameters = param_config

                payload = {}
                payload['param_config'] = json.dumps(param_config)


                # here data has been sent to the data ingestion module
                ingestion_resp = json.loads(requests.post('http://127.0.0.1:5000/ingest', data=payload).text)
                ingestion_resp['dataset'] = ingestion_resp['dataset']
       

                prediction_resp = json.loads(requests.post('http://127.0.0.1:4000/predict', data=ingestion_resp).text)

                return dash.no_update, renderPrediction(prediction_resp, param_config)

            except ValueError as e:
                print(e)
                return 'Error in parsing the json file'
        else:
            return 'Wrong file format, please retry!'
    else:
        return 'Updated failed, please retry'


def renderPrediction(prediction_resp, prediction_parameters):

    timeseries_containers = prediction_resp['timeseries_containers']
    timeseries_containers = pickle.loads(
        base64.b64decode(timeseries_containers))

    global children_for_each_timeseries

    children_for_each_timeseries = [{
        'name': s.timeseries_data.columns[0],
        'children': create_timeseries_dash_children(s, prediction_parameters)
    } for s in timeseries_containers]

    predictionDiv_children = [
        html.H2(prediction_parameters['activity_title']),
        html.H2("Please select the data of interest:"),
        dcc.Dropdown(
            id='timeseries_selector',
            options=[{'label': i['name'], 'value': i['name']}
                     for i in children_for_each_timeseries],
            value='Time-series'
        ), html.Div(id="timeseries_wrapper"), html.Div(dcc.Graph(), style={'display': 'none'})
    ]

    return predictionDiv_children


@ app.callback(
    Output(component_id='timeseries_wrapper',
            component_property='children'),
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
    app.run_server(debug=True, port=3000)