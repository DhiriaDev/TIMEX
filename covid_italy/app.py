import json
import logging
import os
import pickle
import sys
import webbrowser
import dateparser
import numpy

from pandas import read_csv, DataFrame

import timex.data_ingestion
from timex.data_prediction.xcorr import calc_xcorr

from timex.data_ingestion import add_freq, data_selection, add_diff_column
from timex.data_prediction.models.prophet_predictor import FBProphet
from timex.scenario import Scenario
from timex.data_prediction import create_scenarios

log = logging.getLogger(__name__)


def compute():

    param_file_nameJSON = 'configurations/configuration_test_covid19italy.json'

    # Load parameters from config file.
    with open(param_file_nameJSON) as json_file:  # opening the config_file_name
        param_config = json.load(json_file)  # loading the json

    # Logging
    log_level = getattr(logging, param_config["verbose"], None)
    if not isinstance(log_level, int):
        log_level = 0
    # %(name)s for module name
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=log_level, stream=sys.stdout)

    # data ingestion
    log.info(f"Started data ingestion.")
    ingested_data = timex.data_ingestion.ingest_timeseries(param_config)  # ingestion of data

    # data selection
    log.info(f"Started data selection.")
    ingested_data = data_selection(ingested_data, param_config)

    # Custom columns
    log.info(f"Adding custom columns.")
    ingested_data["New cases/tests ratio"] = [100 * (np / tamp) for np, tamp in
                                              zip(ingested_data['Daily cases'], ingested_data['Daily tests'])]

    # data prediction
    scenarios = create_scenarios(ingested_data=ingested_data, param_config=param_config)

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

    max_lags = param_config['xcorr_parameters']['xcorr_max_lags']
    modes = [*param_config['xcorr_parameters']["xcorr_mode"].split(",")]
    try:
        max_threads = param_config['max_threads']
    except KeyError:
        try:
            max_threads = len(os.sched_getaffinity(0))
        except:
            max_threads = 1

    for region in daily_cases_regions.columns:
        scenario_data = daily_cases_regions[[region]]

        model_results = {}

        xcorr = calc_xcorr(region, daily_cases_regions, max_lags, modes)

        log.info(f"Computing univariate prediction for {region}...")
        predictor = FBProphet(param_config, transformation="none")
        prophet_result = predictor.launch_model(scenario_data.copy(), max_threads=max_threads)
        model_results['fbprophet'] = prophet_result
        #
        # predictor = ARIMA(param_config)
        # arima_result = predictor.launch_model(scenario_data.copy())
        # model_results.append(arima_result)

        s = Scenario(scenario_data, model_results, xcorr)
        scenarios.append(s)

        # children_for_each_scenario.append({
        #     'name': region,
        #     'children': create_scenario_children(s, param_config)
        # })

    ####################################################################################################################

    # Save the children; these are the scenarios objects from which a nice Dash page can be built.
    # They can be loaded by "app_load_from_dump.py" to start the app
    # without re-computing all the data.
    with open(f"scenarios.pkl", 'wb') as input_file:
        pickle.dump(scenarios, input_file)


if __name__ == '__main__':
    compute()


    def open_browser():
        webbrowser.open("http://127.0.0.1:8000")


    # Timer(6, open_browser).start()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.system("gunicorn -b 0.0.0.0:8003 app_load_from_dump:server")


