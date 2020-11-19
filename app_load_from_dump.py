import pickle

import dash
import dash_html_components as html
from datetime import datetime, timezone
import dash_core_components as dcc
from dash.dependencies import Input, Output


# Initialize Dash app.
app = dash.Dash(__name__)
server = app.server

now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

# Get children fromfile
with open('children_for_each_scenario.pkl', 'rb') as input_file:
    children_for_each_scenario = pickle.load(input_file)

disclaimer = [html.Div([
    html.H1("COVID-19 pandemic in Italy: monitoring and forecasting", style={'text-align': 'center'}),
    html.Hr(),
    html.Div(html.Img(src=app.get_asset_url('poli.png'), style={'width': 256}), style={'text-align': 'center'}),
    html.H3(
        "Dashboard by the Intelligent Embedded Systems (IES) research group of the Politecnico di Milano, Italy"),
    html.Hr(),
    dcc.Markdown('''
        *Welcome to the monitoring and forecasting dashboard of the Coronavirus (COVID-19) pandemic in Italy provided by the Intelligent Embedded Systems (IES) research group of Politecnico di Milano, Italy.*

        The dashboard relies on *TIMEX*, a Python-based framework for automatic time series analysis developed by the IES research group.

        The dashboard is fed with the [data](https://github.com/pcm-dpc/COVID-19) provided by Italian Civil Protection from Feb. 21 2020. 
        In particular, the following COVID-19 data are considered:
        - **Daily cases**: New positive cases. This is the typical number reported by media.
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
    id='scenario_selector',
    options=[{'label': i['name'], 'value': i['name']} for i in children_for_each_scenario],
    value='Scenario'
), html.Div(id="scenario_wrapper"), html.Div(dcc.Graph(), style={'display': 'none'})]
tree = html.Div(children=disclaimer, style={'width': '80%', 'margin': 'auto'})

app.layout = tree


@app.callback(
    Output(component_id='scenario_wrapper', component_property='children'),
    [Input(component_id='scenario_selector', component_property='value')]
)
def update_scenario_wrapper(input_value):
    try:
        children = next(x['children'] for x in children_for_each_scenario if x['name'] == input_value)
    except StopIteration:
        return html.Div(style={'padding': 200})

    return children
