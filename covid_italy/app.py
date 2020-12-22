import json
import os
import pickle
import sys
import webbrowser
import logging

import dash_html_components as html
import dateparser
import numpy
from pandas import read_csv, DataFrame
import pandas as pd

from timex.data_ingestion import data_ingestion
from timex.data_ingestion.data_ingestion import add_freq
from timex.data_prediction.data_prediction import calc_xcorr
from timex.data_prediction.prophet_predictor import FBProphet
from timex.data_preparation.data_preparation import data_selection, add_diff_column
from timex.data_visualization.data_visualization import create_scenario_children, line_plot_multiIndex
from timex.scenario.scenario import Scenario

log = logging.getLogger(__name__)


def create_children():

    param_file_nameJSON = 'configurations/configuration_test_covid19italy.json'

    # Load parameters from config file.
    with open(param_file_nameJSON) as json_file:  # opening the config_file_name
        param_config = json.load(json_file)  # loading the json

    # Logging
    log_level = getattr(logging, param_config["verbose"], None)
    if not isinstance(log_level, int):
        log_level = 0
    logging.basicConfig(format='%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s', level=log_level,
                        stream=sys.stdout)

    # data ingestion
    log.info(f"Started data ingestion.")
    ingested_data = data_ingestion.data_ingestion(param_config)  # ingestion of data

    # data selection
    log.info(f"Started data selection.")
    ingested_data = data_selection(ingested_data, param_config)

    # Custom columns
    log.info(f"Adding custom columns.")
    ingested_data["New cases/tests ratio"] = [100*(np/tamp) for np, tamp in zip(ingested_data['Daily cases'], ingested_data['Daily tests'])]

    # data prediction
    log.info(f"Computing the cross-correlation...")
    max_lags = param_config['model_parameters']['xcorr_max_lags']
    modes = [*param_config['model_parameters']["xcorr_mode"].split(",")]
    try:
        max_threads = param_config['max_threads']
    except KeyError:
        try:
            max_threads = len(os.sched_getaffinity(0))
        except:
            max_threads = 1

    log.info(f"Started the prediction with univariate models.")
    log.debug(f"Training will use {max_threads} threads...")
    columns = ingested_data.columns
    scenarios = []
    for col in columns:
        scenario_data = ingested_data[[col]]
        model_results = {}

        xcorr = calc_xcorr(col, ingested_data, max_lags, modes)

        log.info(f"Computing univariate prediction for {col}...")
        predictor = FBProphet(param_config)
        prophet_result = predictor.launch_model(scenario_data.copy(), max_threads=max_threads)
        model_results['fbprophet'] = prophet_result
        #
        # predictor = ARIMA(param_config)
        # arima_result = predictor.launch_model(scenario_data.copy())
        # model_results.append(arima_result)

        scenarios.append(
            Scenario(scenario_data, model_results, xcorr)
        )

    # data visualization
    children_for_each_scenario = [{
        'name': s.scenario_data.columns[0],
        'children': create_scenario_children(s, param_config)
    } for s in scenarios]

    ####################################################################################################################
    # Custom scenario #########
    log.info(f"Computing the custom scenarios.")

    regions = read_csv("https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv",
                       header=0, index_col=0, usecols=['data', 'denominazione_regione', 'nuovi_positivi', 'tamponi'])
    regions.reset_index(inplace=True)
    regions['data'] = regions['data'].apply(lambda x: dateparser.parse(x))
    regions.set_index(['data', 'denominazione_regione'], inplace=True, drop=True)

    regions = add_diff_column(regions, ['tamponi'], group_by='denominazione_regione')

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

    # Append "Regioni" scenario
    children_for_each_scenario.append({'name': 'Regions', 'children': regions_children})

    # Prediction of "New daily cases" for every region
    # We also want to plot cross-correlation with other regions.
    # So, create a dataFrame with only daily cases and regions as columns.
    regions_names = regions.index.get_level_values(1).unique()
    regions_names = regions_names.sort_values()

    datas = regions.index.get_level_values(0).unique().to_list()
    datas = datas[1:]  # Abruzzo is missing the first day.

    cols = regions_names.to_list()
    cols = ['data'] + cols

    daily_cases_regions = DataFrame(columns=cols, dtype=numpy.float64)
    daily_cases_regions['data'] = datas

    daily_cases_regions.set_index(['data'], inplace=True, drop=True)

    for col in daily_cases_regions.columns:
        for i in daily_cases_regions.index:
            daily_cases_regions.loc[i][col] = regions.loc[i, col]['Daily cases']

    daily_cases_regions = add_freq(daily_cases_regions, 'D')

    for region in daily_cases_regions.columns:
        scenario_data = daily_cases_regions[[region]]

        model_results = {}

        xcorr = calc_xcorr(region, daily_cases_regions, max_lags, modes)

        log.info(f"Computing univariate prediction for {region}...")
        predictor = FBProphet(param_config)
        prophet_result = predictor.launch_model(scenario_data.copy(), max_threads=max_threads)
        model_results['fbprophet'] = prophet_result
        #
        # predictor = ARIMA(param_config)
        # arima_result = predictor.launch_model(scenario_data.copy())
        # model_results.append(arima_result)

        s = Scenario(scenario_data, model_results, xcorr)

        children_for_each_scenario.append({
            'name': region,
            'children': create_scenario_children(s, param_config)
        })

    ####################################################################################################################

    # Save the children; these are the plots relatives to all the scenarios.
    # They can be loaded by "app_load_from_dump.py" to start the app
    # without re-computing all the plots.
    curr_dirr = os.path.dirname(os.path.abspath(__file__))
    filename = 'children_for_each_scenario.pkl'
    log.info(f"Saving the computed Dash children to {curr_dirr}/{filename}...")
    with open(f"{curr_dirr}/{filename}", 'wb') as input_file:
        pickle.dump(children_for_each_scenario, input_file)


if __name__ == '__main__':

    create_children()

    def open_browser():
        webbrowser.open("http://127.0.0.1:8000")

    # Timer(6, open_browser).start()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.system("gunicorn -b 0.0.0.0:8000 app_load_from_dump:server")

