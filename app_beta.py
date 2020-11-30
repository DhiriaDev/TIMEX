import json
import logging
import os
import pickle
import sys
import webbrowser
from functools import reduce

import numpy
import pandas as pd
from pandas import read_csv, DataFrame
import dash_html_components as html


from timex.data_ingestion import data_ingestion
from timex.data_ingestion.data_ingestion import add_freq
from timex.data_prediction.data_prediction import calc_all_xcorr, calc_xcorr
from timex.utils.utils import prepare_extra_regressor
from timex.data_prediction.prophet_predictor import FBProphet
from timex.data_preparation.data_preparation import data_selection, add_diff_column
from timex.data_visualization.data_visualization import create_scenario_children, line_plot_multiIndex
from timex.scenario.scenario import Scenario

log = logging.getLogger(__name__)


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

    # Logging
    log_level = getattr(logging, param_config["verbose"], None)
    if not isinstance(log_level, int):
        log_level = 0
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=log_level,
                        stream=sys.stdout)

    # data ingestion
    log.info(f"Started data ingestion.")
    ingested_data = data_ingestion.data_ingestion(param_config)  # ingestion of data

    # data selection
    log.info(f"Started data selection.")
    ingested_data = data_selection(ingested_data, param_config)

    # Custom columns
    log.info(f"Adding custom columns.")
    ingested_data["New cases/tests ratio"] = [100 * (np / tamp) for np, tamp in
                                              zip(ingested_data['Daily cases'], ingested_data['Daily tests'])]

    # Calculate the cross-correlation.
    log.info(f"Computing the cross-correlation...")
    xcorr_max_lags = param_config['model_parameters']['xcorr_max_lags']
    xcorr_modes = [*param_config['model_parameters']["xcorr_mode"].split(",")]
    xcorr_threshold = param_config['model_parameters']['xcorr_extra_regressor_threshold']
    xcorr_mode_target = param_config['model_parameters']['xcorr_mode_target']
    main_accuracy_estimator = param_config['model_parameters']['main_accuracy_estimator']

    total_xcorr = calc_all_xcorr(ingested_data=ingested_data, max_lags=xcorr_max_lags, modes=xcorr_modes)

    # Predictions without extra-regressors.
    log.info(f"Started the prediction with univariate models.")
    columns = ingested_data.columns
    scenarios = []
    for col in columns:
        scenario_data = ingested_data[[col]]
        model_results = []

        xcorr = total_xcorr[col]

        log.info(f"Computing univariate prediction for {col}...")
        predictor = FBProphet(param_config)
        prophet_result = predictor.launch_model(scenario_data.copy())
        model_results.append(prophet_result)

        # predictor = ARIMA(param_config)
        # arima_result = predictor.launch_model(scenario_data.copy())
        # model_results.append(arima_result)

        scenarios.append(
            Scenario(scenario_data, model_results, ingested_data, xcorr)
        )

    # Prediction with extra regressors.
    log.info(f"Starting the prediction with extra-regressors.")
    best_forecasts_found = 0
    while best_forecasts_found != len(columns):
        log.info(f"-> Found the optimal prediction for only {best_forecasts_found}")
        best_forecasts_found = 0

        for col in columns:
            local_xcorr = total_xcorr[col][xcorr_mode_target]

            useful_extra_regressors = []

            for extra_regressor in local_xcorr.columns:
                # Look only in correlation with future lags.
                index_of_max = local_xcorr[extra_regressor].abs().idxmax()
                corr = local_xcorr.loc[index_of_max, extra_regressor]
                if abs(corr) > xcorr_threshold and index_of_max >= 0:
                    log.debug(f"Found a possible extra-regressor for {col}: {extra_regressor} at lag {index_of_max}")
                    useful_extra_regressors.append(extra_regressor)

            if len(useful_extra_regressors) == 0:
                log.debug(f"No useful extra-regressor found for {col}: skipping...")
                best_forecasts_found += 1
            else:
                log.info(f"Found useful extra-regressors. Prepare them and re-compute the prediction for {col}")
                useful_extra_regressors = [
                    prepare_extra_regressor(next(filter(lambda x: x.scenario_data.columns[0] == s, scenarios)),
                                            main_accuracy_estimator) for s in useful_extra_regressors]

                useful_extra_regressors = reduce(lambda x, y: x.join(y), useful_extra_regressors)

                scenario_data = ingested_data[[col]]
                model_results = []

                predictor = FBProphet(param_config)
                prophet_result = predictor.launch_model(scenario_data.copy(), extra_regressors=useful_extra_regressors.copy())
                model_results.append(prophet_result)

                old_this_scenario = next(filter(lambda x: x.scenario_data.columns[0] == col, scenarios))
                old_errors = [x.testing_performances.MAE for x in old_this_scenario.models[0].results]
                min_old_error = min(old_errors)
                min_new_error = min([x.testing_performances.MAE for x in prophet_result.results])

                if min_new_error < min_old_error:
                    log.info(f"Obtained a better error: {min_new_error} vs old {min_old_error}")
                    new_scenario = Scenario(scenario_data, model_results, ingested_data, total_xcorr[col])
                    scenarios = [new_scenario if x.scenario_data.columns[0] == col else x for x in scenarios]
                else:
                    log.info(f"No improvements.")
                    best_forecasts_found += 1

    log.info(f"Found the optimal prediction for all the {best_forecasts_found} scenarios!")

    # data visualization
    children_for_each_scenario = [{
        'name': s.scenario_data.columns[0],
        'children': create_scenario_children(s, param_config)
    } for s in scenarios]

    ####################################################################################################################
    # Custom scenario #########

    # regions = read_csv("https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv",
    #                    header=0, index_col=0, usecols=['data', 'denominazione_regione', 'nuovi_positivi', 'tamponi'])
    # regions.reset_index(inplace=True)
    # regions['data'] = pd.to_datetime(regions['data'], format=param_config['input_parameters']['datetime_format'])
    # regions.set_index(['data', 'denominazione_regione'], inplace=True, drop=True)
    #
    # regions = add_diff_column(regions, ['tamponi'], group_by='denominazione_regione')
    #
    # regions.rename(columns={'nuovi_positivi': 'Daily cases', 'tamponi': 'Tests',
    #                         "tamponi_diff": "Daily tests"}, inplace=True)
    #
    # regions["New cases/tests ratio"] = [100*(ndc/tamp) if tamp > ndc > 0 else "nan" for ndc, tamp in
    #                                     zip(regions['Daily cases'], regions['Daily tests'])]
    #
    # regions_children = [
    #     html.H2(children='Regions' + " analysis", id='Regions'),
    #     html.Em("You can select a specific region by doucle-clicking on its label (in the right list); clicking "
    #             "on other regions, you can select only few of them."),
    #     html.H3("Data visualization"),
    #     line_plot_multiIndex(regions[['Daily cases']]),
    #     line_plot_multiIndex(regions[['Daily tests']]),
    #     line_plot_multiIndex(regions[['New cases/tests ratio']])
    # ]
    #
    # # Append "Regioni" scenario
    # children_for_each_scenario.append({'name': 'Regions', 'children': regions_children})
    #
    # # Prediction of "New daily cases" for every region
    # # We also want to plot cross-correlation with other regions.
    # # So, create a dataFrame with only daily cases and regions as columns.
    # regions_names = regions.index.get_level_values(1).unique()
    # regions_names = regions_names.sort_values()
    #
    # datas = regions.index.get_level_values(0).unique().to_list()
    # datas = datas[1:]  # Abruzzo is missing the first day.
    #
    # cols = regions_names.to_list()
    # cols = ['data'] + cols
    #
    # daily_cases_regions = DataFrame(columns=cols, dtype=numpy.float64)
    # daily_cases_regions['data'] = datas
    #
    # daily_cases_regions['data'] = pd.to_datetime(daily_cases_regions['data'], format=param_config['input_parameters']['datetime_format'])
    # daily_cases_regions.set_index(['data'], inplace=True, drop=True)
    #
    # for col in daily_cases_regions.columns:
    #     for i in daily_cases_regions.index:
    #         daily_cases_regions.loc[i][col] = regions.loc[i, col]['Daily cases']
    #
    # daily_cases_regions = add_freq(daily_cases_regions, 'D')
    #
    # for region in daily_cases_regions.columns:
    #     scenario_data = daily_cases_regions[[region]]
    #
    #     model_results = []
    #
    #     print('-> Calculate the cross-correlation...')
    #     xcorr = calc_xcorr(region, daily_cases_regions, xcorr_max_lags, xcorr_modes)
    #
    #     print('-> PREDICTION FOR ' + str(region))
    #     predictor = FBProphet(param_config)
    #     prophet_result = predictor.launch_model(scenario_data.copy())
    #     model_results.append(prophet_result)
    #     #
    #     # predictor = ARIMA(param_config)
    #     # arima_result = predictor.launch_model(scenario_data.copy())
    #     # model_results.append(arima_result)
    #
    #     s = Scenario(scenario_data, model_results, daily_cases_regions, xcorr)
    #
    #     children_for_each_scenario.append({
    #         'name': region,
    #         'children': create_scenario_children(s, param_config)
    #     })

    ####################################################################################################################

    # Save the children; these are the plots relatives to all the scenarios.
    # They can be loaded by "app_load_from_dump.py" to start the app
    # without re-computing all the plots.
    filename = 'children_for_each_scenario_beta.pkl'
    log.info(f"Saving the computed Dash children to {filename}...")
    with open(filename, 'wb') as input_file:
        pickle.dump(children_for_each_scenario, input_file)


if __name__ == '__main__':
    create_children()


    def open_browser():
        webbrowser.open("http://127.0.0.1:8000")


    # Timer(6, open_browser).start()
    os.system("gunicorn -b 0.0.0.0:8002 app_load_from_dump_beta:server")


