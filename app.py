import json
import pickle
import webbrowser
from threading import Timer

import dash
import dash_html_components as html

from timex.data_ingestion import data_ingestion
from timex.data_prediction.arima_predictor import ARIMA
from timex.data_prediction.prophet_predictor import FBProphet
from timex.data_preparation.data_preparation import data_selection, add_diff_column
from timex.data_visualization.data_visualization import create_dash_children
from timex.scenario.scenario import Scenario


example = "covid19italy"

if example == "covid19italy":
    param_file_nameJSON = 'demo_configurations/configuration_test_covid19italy.json'
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
    target = param_config["input_parameters"]["add_diff_column"]
    name = target + "_diff"
    ingested_data = add_diff_column(ingested_data, target, name, "yes")

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
    predictor = ARIMA(param_config)
    arima_result = predictor.launch_model(scenario_data.copy())
    model_results.append(arima_result)

    scenarios.append(
        Scenario(scenario_data, model_results)
    )

# data visualization
children = create_dash_children(scenarios, param_config)

# # Initialize Dash app.
app = dash.Dash(__name__)
server = app.server

print("Serving the layout...")
app.layout = html.Div(children=children)


# with open('children.pkl', 'wb') as input_file:
#     pickle.dump(children, input_file)


def open_browser():
    webbrowser.open("http://127.0.0.1:8050")


Timer(1, open_browser).start()


if __name__ == '__main__':
    app.run_server()
