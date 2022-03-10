import logging
import sys

from flask import Flask, request
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import json
import pickle
import base64
from matplotlib.font_manager import json_dump
import requests
import io

from data_visualization.functions import create_timeseries_dash_children
from utils import TimeSeriesContainer


timex_manager_address='http://127.0.0.1:6000/predict'


# -----------SERVER INIT-----------------------

server = Flask(__name__)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
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

#--------LANDING PAGE-------------

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

    if(config_file is not None and filename is not None and 'json' in filename):

        encoding = 'utf-8'

        # Check the validity of the input file
        try:
            content_type, content_string = config_file.split(',')

            param_config = json.load(
                io.StringIO(((base64.b64decode(content_string)).decode(encoding))))


        except ValueError as e:
            print(e)
            return 'Error in parsing the json file'

        try:
            logger.info('Contacting the timex_manager')

            prediction_resp = json.loads(requests.post(
                timex_manager_address, 
                data={ 'param_config' : json.dumps(param_config)}).text)
            logger.info('Predictions received')

        except ValueError as e:
            print(e)
            return 'Error in contacting the prediction module'

            
        return dash.no_update, renderPrediction(prediction_resp, param_config)

    else:
        return 'Updated failed, please retry', 400


def renderPrediction(prediction_resp, prediction_parameters):
    
    logger.info('Results rendering started.')
    best_model = pickle.loads(base64.b64decode(prediction_resp['best_model']))
    
    #--------PREDICTION RENDERING SECTION----------
    # even if we have a single result, it is better to work with lists for better scalability
    children_for_each_timeseries = []
    children_for_each_timeseries.append({
        'name': best_model.timeseries_data.columns[0],
        'children': create_timeseries_dash_children(best_model, prediction_parameters)
    })

    predictionDiv_children = [
        html.H2(prediction_parameters['activity_title']),
        html.H2("Please select the data of interest:"),
        dcc.Dropdown(
            id='timeseries_selector',
            options=[{'label': i['name'], 'value': i['name']}
                     for i in children_for_each_timeseries],
            value='Time-series'
        ), html.Div(id="timeseries_wrapper"), html.Div(dcc.Graph(), style={'display': 'none'}),
        dcc.Store(id='children_for_each_timeseries', data=children_for_each_timeseries)
    ]

    return predictionDiv_children


@ app.callback(
    Output(component_id='timeseries_wrapper',
           component_property='children'),
    Input(component_id='timeseries_selector', component_property='value'),
    Input(component_id='children_for_each_timeseries', component_property='data'),

    prevent_initial_call=True
)
def update_timeseries_wrapper(input_value, children_for_each_timeseries):
    try:
        children = next(
            x['children'] for x in children_for_each_timeseries if x['name'] == input_value)
    except StopIteration:
        return html.Div(style={'padding': 200})

    return children


app_address = '127.0.0.1'
app_docker_address='0.0.0.0'
app_port = 5000

if __name__ == '__main__':
    app.run_server(host=app_address, port=app_port ,debug=True)
