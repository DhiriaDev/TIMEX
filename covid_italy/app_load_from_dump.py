import json
import logging
import os
import pickle

import dash
import dash_html_components as html
from datetime import datetime, timezone
import dash_core_components as dcc
import dateparser
from dash.dependencies import Input, Output
from pandas import read_csv

from timex.data_ingestion import add_diff_columns
from timex.data_visualization.functions import create_timeseries_dash_children, line_plot_multiIndex

log = logging.getLogger(__name__)


param_file_nameJSON = 'configurations/configuration_test_covid19italy.json'

# Load parameters from config file.
with open(param_file_nameJSON) as json_file:  # opening the config_file_name
    param_config = json.load(json_file)  # loading the json

# Load containers dump.
with open(f"containers.pkl", 'rb') as input_file:
    timeseries_containers = pickle.load(input_file)

# Data visualization
children_for_each_timeseries = [{
    'name': s.timeseries_data.columns[0],
    'children': create_timeseries_dash_children(s, param_config)
} for s in timeseries_containers]

#######################################################################################################################
#### CUSTOM TIME-SERIES ##
#######################################################################################################################
regions = read_csv("https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv",
                   header=0, index_col=0, usecols=['data', 'denominazione_regione', 'nuovi_positivi', 'tamponi'])
regions.reset_index(inplace=True)
regions['data'] = regions['data'].apply(lambda x: dateparser.parse(x))
regions.set_index(['data', 'denominazione_regione'], inplace=True, drop=True)

regions = add_diff_columns(regions, ['tamponi'], group_by='denominazione_regione')

regions.rename(columns={'nuovi_positivi': 'Daily cases', 'tamponi': 'Tests',
                        "tamponi_diff": "Daily tests"}, inplace=True)

regions["New cases/tests ratio"] = [100*(ndc/tamp) if tamp > ndc > 0 else "nan" for ndc, tamp in
                                    zip(regions['Daily cases'], regions['Daily tests'])]

regions_children = [
    html.H2(children='Regions' + " analysis", id='Regions'),
    html.Em("You can select a specific region by doucle-clicking on its label (in the right list); clicking "
            "on other regions, you can select only few of them."),
    html.H3("Data visualization"),
    line_plot_multiIndex(regions[['Daily cases']]),
    line_plot_multiIndex(regions[['Daily tests']]),
    line_plot_multiIndex(regions[['New cases/tests ratio']])
]


children_for_each_timeseries.append({'name': 'Regions', 'children': regions_children})

# Initialize Dash app.
app = dash.Dash(__name__)
server = app.server

now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

disclaimer = [html.Div([
    html.H1("COVID-19 pandemic in Italy: monitoring and forecasting", style={'text-align': 'center'}),
    html.Hr(),
    html.Div(html.Img(src=app.get_asset_url('poli.png'), style={'width': 256}), style={'text-align': 'center'}),
    html.H4(
        "Dashboard by the Intelligent Embedded Systems (IES) research group of the Politecnico di Milano, Italy",
        style={'text-align': 'center', 'top-margin': '25px'}),
    html.Hr(),
    dcc.Markdown('''
        *Welcome to the monitoring and forecasting dashboard of the Coronavirus (COVID-19) pandemic in Italy provided by the Intelligent Embedded Systems (IES) research group of Politecnico di Milano, Italy.*

        The dashboard relies on *TIMEX*, a Python-based framework for automatic time series analysis developed by the IES research group.

        The dashboard is fed with the [data](https://github.com/pcm-dpc/COVID-19) provided by Italian Civil Protection from Feb. 21 2020. 
        In particular, the following COVID-19 data are considered:
        - **Daily cases**: New positive cases. This is the typical number reported by media.
        - **Total positives**: The total number of persons positive to Covid, in that time instant.
        - **Total positives variation**: Daily variation of the number of persons positive to Covid.
        - **Total intensive care**: Total number of patients in intensive care.
        - **Total hospitalized**: Total number of patients in hospitals.
        - **Total deaths**: Total number of deaths due to Covid-19.
        - **Daily intensive care**: New intensive care patients.
        - **Daily hospitalized**: New patients in hospital.
        - **Daily deaths**: Daily deaths.
        - **Daily tests**: Daily tests.
        - **Positive cases/test ratio**: Daily ratio of positive tests.
        - **Regions**: Mixed information about single regions.

        You can select the data to be visualized from the selector at the bottom of the page.

        For suggestions and questions contact:
        - Prof. Manuel Roveri - manuel.roveri (at) polimi.it
        - Ing. Alessandro Falcetta - alessandro.falcetta (at) mail.polimi.it

        *DISCLAIMER: The information on this site is not intended or implied to be a substitute for professional medical advice, diagnosis or treatment. Contents, including text, graphics, images and information, presented in or available on this web site are meant to help in advancing the understanding of the virus and informing the public.
         Information is subject to change without notice. You are encouraged to confirm any information obtained from or through this web site with other sources.*
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


