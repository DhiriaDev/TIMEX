import json
import pickle
import webbrowser
from threading import Timer

import dash_html_components as html
from pandas import read_csv
import pandas as pd

import app_load_from_dump
from timex.data_ingestion import data_ingestion
from timex.data_ingestion.data_ingestion import add_freq
from timex.data_prediction.arima_predictor import ARIMA
from timex.data_prediction.prophet_predictor import FBProphet
from timex.data_preparation.data_preparation import data_selection, add_diff_column
from timex.data_visualization.data_visualization import create_scenario_children, line_plot_multiIndex
from timex.scenario.scenario import Scenario


def create_children():
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

    ####################################################################################################################
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
        line_plot_multiIndex(regions[['Daily tests difference']]),
        line_plot_multiIndex(regions[['Ratio New cases/tests']])
    ]

    # Append "Regioni" scenario
    children_for_each_scenario.append({'name': 'Regions', 'children': regions_children})

    # Prediction of "New daily cases" for every region
    regions_names = regions.index.get_level_values(1).unique()
    regions_names = regions_names.sort_values()

    for region in regions_names:
        region_data = regions.loc[(regions.index.get_level_values('denominazione_regione') == region)]
        region_data.reset_index(inplace=True)
        region_data.set_index('data', inplace=True)
        region_data = add_freq(region_data, 'D')
        region_data = region_data[['New daily cases']]

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

    ####################################################################################################################

    # Save the children; these are the plots relatives to all the scenarios.
    # They can be loaded by "app_load_from_dump.py" to start the app
    # without re-computing all the plots.
    with open('children_for_each_scenario.pkl', 'wb') as input_file:
        pickle.dump(children_for_each_scenario, input_file)


if __name__ == '__main__':
    create_children()

    port = 10000

    def open_browser():
        webbrowser.open("http://127.0.0.1:" + str(port))


    Timer(1, open_browser).start()
    app_load_from_dump.app.run_server(port=port)

