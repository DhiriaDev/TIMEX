
from json.decoder import JSONDecoder
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.html.Label import Label
from pandas._config import config

# Connect to main app.py file
from app import app

import json
import pandas as pd
import base64
import io

uploadStyle = {
    'width': '40%',
    'height': '60px',
    'lineHeight': '60px',
    'borderWidth': '1px',
    'borderStyle': 'dashed',
    'borderRadius': '5px',
    'textAlign': 'center'
}


app.layout = html.Div([
    html.H1('Welcome to Timex, a machine learning time series forecasting a.a.S.'),
    html.Br(),

    html.Div(
        children=[
            html.H3('Please insert your dataset in a csv and the configuration json'),


            html.Label(children='Upload here your files: ',
                       htmlFor='uploaded'),
            dcc.Upload(html.A('Select a File'), id='uploaded',
                       style=uploadStyle, multiple=True),
        ]
    ),

    html.Hr(),
    html.H3(id='error_sentence', style={"color": "red"}),
    html.Div(id='forecasted_output')
])


@ app.callback(
    Output(component_id='error_sentence', component_property='children'),
    Output(component_id='forecasted_output', component_property='children'),
    Input(component_id='uploaded', component_property='contents'),
    State(component_id='uploaded', component_property='filename'),
    # it prevents the callback from being executed when the page is refreshed
    prevent_initial_call=True
)
def compute_forecasting(config_files, filenames):

    if(config_files is not None and filenames is not None and len(config_files) == 2 and len(filenames) == 2):

        encoding = 'utf-8'
        for i in range(0, len(filenames)):
            if 'csv' in filenames[i]:

                try:
                    content_type, content_string = config_files[i].split(',')

                    dataset = pd.read_csv(
                        io.StringIO((base64.b64decode(content_string)).decode(encoding)))

                except ValueError as e:
                    print(e)
                    return 'Error in parsing the csv file', dash.no_update

            if 'json' in filenames[i]:
                try:
                    content_type, content_string = config_files[i].split(',')

                    config = json.load(
                        io.StringIO(((base64.b64decode(content_string)).decode(encoding))))
                        
                except ValueError as e:
                    print(e)
                    return 'Error in parsing the json file', dash.no_update

        if(dataset is None or config is None):
            return 'Updated failed, please retry', dash.no_update
        else:
            return 'Successfully updated all the files ! ', dash.no_update

    else:
        return 'Updated failed, please retry', dash.no_update


if __name__ == '__main__':
    app.run_server(debug=True)
