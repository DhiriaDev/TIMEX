import json
import pickle
import webbrowser
from datetime import datetime, timezone
from threading import Timer

import dash
import dash_html_components as html
import dash_core_components as dcc
from pandas import read_csv
import pandas as pd

from timex.data_ingestion import data_ingestion
from timex.data_ingestion.data_ingestion import add_freq
from timex.data_prediction.arima_predictor import ARIMA
from timex.data_prediction.prophet_predictor import FBProphet
from timex.data_preparation.data_preparation import data_selection, add_diff_column
from timex.data_visualization.data_visualization import create_scenario_children, line_plot_multiIndex
from timex.scenario.scenario import Scenario
from dash.dependencies import Input, Output


example = "covid19italy"

if example == "covid19italy":
    param_file_nameJSON = 'demo_configurations/configuration_test_covid19italy.json'
# elif example == "covid19italyregions":
#     param_file_nameJSON = 'demo_configurations/configuration_test_covid19italy_regions.json'
elif example == "airlines":
    param_file_nameJSON = 'demo_configurations/configuration_test_airlines.json'
elif example == "covid19switzerland":
    param_file_nameJSON = 'demo_configurations/configuration_test_covid19switzerland.json'
elif example == "temperatures":
    param_file_nameJSON = 'demo_configurations/configuration_test_daily_min_temperatures.json'
else:
    exit()

# Load parameters from config file.
with open(param_file_nameJSON) as json_file:  # opening the config_file_name
    param_config = json.load(json_file)  # loading the json

# data ingestion
print('-> INGESTION')
ingested_data = data_ingestion.data_ingestion(param_config)  # ingestion of data

# data selection
print('-> SELECTION')
ingested_data = data_selection(ingested_data, param_config)

if "add_diff_column" in param_config["input_parameters"]:
    print('-> ADD DIFF COLUMN')
    targets = list(param_config["input_parameters"]["add_diff_column"].split(','))
    ingested_data = add_diff_column(ingested_data, targets, verbose="yes")

# Rename columns
mappings = param_config["input_parameters"]["scenarios_names"]
ingested_data.rename(columns=mappings, inplace=True)

# Custom columns
ingested_data["Ratio New cases/tests"] = [100*(np/tamp) for np, tamp in zip(ingested_data['New daily cases'], ingested_data['Daily tests difference'])]

# data prediction
columns = ingested_data.columns
scenarios = []
for col in columns:
    scenario_data = ingested_data[[col]]
    model_results = []

    print('-> PREDICTION FOR ' + str(col))
    predictor = FBProphet(param_config)
    prophet_result = predictor.launch_model(scenario_data.copy())
    model_results.append(prophet_result)
    #
    # predictor = ARIMA(param_config)
    # arima_result = predictor.launch_model(scenario_data.copy())
    # model_results.append(arima_result)

    scenarios.append(
        Scenario(scenario_data, model_results)
    )

# data visualization
children_for_each_scenario = [{
    'name': s.ingested_data.columns[0],
    'children': create_scenario_children(s, param_config)
} for s in scenarios]

#######################################################################################################################
# Custom scenario #########
regions = read_csv("https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv",
                   header=0, index_col=0, usecols=['data', 'denominazione_regione', 'nuovi_positivi', 'tamponi'])
regions.reset_index(inplace=True)
regions['data'] = pd.to_datetime(regions['data'], format=param_config['input_parameters']['datetime_format'])
regions.set_index(['data', 'denominazione_regione'], inplace=True, drop=True)

regions = add_diff_column(regions, ['tamponi'], group_by='denominazione_regione')

regions.rename(columns={'nuovi_positivi': 'New daily cases', 'tamponi': 'Tests',
                        "tamponi_diff": "Daily tests difference"}, inplace=True)

regions["Ratio New cases/tests"] = [100*(ndc/tamp) if tamp > ndc > 0 else "nan" for ndc, tamp in
                                    zip(regions['New daily cases'], regions['Daily tests difference'])]

regions_children = [
    html.H2(children='Regions' + " analysis", id='Regions'),
    html.Em("You can select a specific region by doucle-clicking on its label (in the right list); clicking "
            "on other regions, you can select only few of them."),
    html.H3("Data visualization"),
    line_plot_multiIndex(regions[['New daily cases']]),
    line_plot_multiIndex(regions[['Ratio New cases/tests']])
]

# Append "Regioni" scenario
children_for_each_scenario.append({'name': 'Regions', 'children': regions_children})

regions_scenarios = []

# Prediction of "New daily cases" for every region
for region in regions.index.get_level_values(1).unique():
    region_data = regions.loc[(regions.index.get_level_values('denominazione_regione') == region)]
    region_data.reset_index(inplace=True)
    region_data.set_index('data', inplace=True)
    region_data = add_freq(region_data, 'D')
    region_data = region_data[['New daily cases']]

    # region_data.drop(columns=['denominazione_regione'], inplace=True)

    model_results = []

    print('-> PREDICTION FOR ' + str(region))
    predictor = FBProphet(param_config)
    prophet_result = predictor.launch_model(region_data.copy())
    model_results.append(prophet_result)
    #
    # predictor = ARIMA(param_config)
    # arima_result = predictor.launch_model(scenario_data.copy())
    # model_results.append(arima_result)

    s = Scenario(region_data, model_results)

    children_for_each_scenario.append({
        'name': region,
        'children': create_scenario_children(s, param_config)
    })

    regions_scenarios.append(s)

#######################################################################################################################

# Initialize Dash app.
app = dash.Dash(__name__)
server = app.server

now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

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
        In particular, the following COVID-19 Data are considered:
        - **New daily cases**: New cases found in that day. This is the common number reported by media.
        - **Total intensive care**: Total number of patients in intensive care.
        - **Total hospitalisations**: Total number of patients in hospitals.
        - **Total deaths**: Total number of deaths due to Covid-19.
        - **Daily intensive care difference**: Difference, w.r.t the previous day, in the number of intensive care patients.
        - **Daily hospitalisations difference**: Difference, w.r.t the previous day, in the number of hospitalisations.
        - **Daily deaths difference**: Difference, w.r.t the previous day, in the number of deaths.
        - **Ratio New cases/tests**: Daily ratio of positive tests.
        - **Regions**: Mixed information about single regions.

        You can select the visualized data from the selector at the bottom of the page.

        For suggestions and questions contact:
        - Prof. Manuel Roveri - manuel.roveri (at) polimi.it
        - Ing. Alessandro Falcetta - alessandro.falcetta (at) mail.polimi.it

        *DISCLAIMER: The information on this site is not intended or implied to be a substitute for professional medical advice, diagnosis or treatment. All content, including text, graphics, images and information, contained on or available through this web site is for general information purposes only.
        We make no representation and assume no responsibility for the accuracy of information contained on or available through this web site, and such information is subject to change without notice. You are encouraged to confirm any information obtained from or through this web site with other sources.*
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


# Save the children; these are the plots relatives to all the scenarios.
# They can be loaded by "app_load_from_dump.py" to start the app
# without re-computing all the plots.
with open('children_for_each_scenario.pkl', 'wb') as input_file:
    pickle.dump(children_for_each_scenario, input_file)


def open_browser():
    webbrowser.open("http://127.0.0.1:8050")


Timer(1, open_browser).start()


if __name__ == '__main__':
    app.run_server()
