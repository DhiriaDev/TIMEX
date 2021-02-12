import logging
import os
import pickle
from collections import defaultdict
from functools import reduce
from typing import Tuple

import dateparser
from pandas import DataFrame
import pandas as pd
from timex.data_prediction.mockup_predictor import MockUpModel

from timex.data_ingestion import ingest_additional_regressors
from timex.data_prediction.neuralprophet_predictor import NeuralProphetModel

from timex.data_prediction.arima_predictor import ARIMA
from timex.data_prediction.data_prediction import calc_all_xcorr, PredictionModel
from timex.data_prediction.lstm_predictor import LSTM_model
from timex.data_prediction.prophet_predictor import FBProphet
from timex.scenario.scenario import Scenario

log = logging.getLogger(__name__)


def prepare_extra_regressor(scenario: Scenario, model: str) -> DataFrame:
    """
    This function receives a Scenario object, which includes the time-series historical data and the various predictions
    for the future.

    The best prediction for the model 'model' is taken and appended to the original time-series, in order to obtain a
    DataFrame with the original time series and the best possible prediction.

    The resulting DataFrame is returned.

    Parameters
    ----------
    scenario : Scenario
    Scenario from which an extra-regressor should be extracted.

    model
    The model from which get the best available prediction.

    Returns
    -------
    df : DataFrame
    DataFrame with the length of the original time-series + prediction lags.
    """
    name = scenario.scenario_data.columns[0]
    best_prediction = scenario.models[model].best_prediction

    original_ts = scenario.scenario_data
    f = best_prediction.loc[:, ['yhat']]
    f.rename(columns={'yhat': name}, inplace=True)

    best_entire_forecast = original_ts.combine_first(f)

    return best_entire_forecast


def get_best_univariate_predictions(ingested_data: DataFrame, param_config: dict, total_xcorr: dict) -> \
        Tuple[dict, list]:
    """
    Compute, for every column in ingested_data (every "scenario") the best univariate prediction possible.
    This is done using the models specified in param_config and testing the effect of the different transformations
    specified in param_config.

    Parameters
    ----------
    ingested_data : DataFrame
    param_config : TIMEX configuration dictionary
    total_xcorr : dict

    Returns
    -------
    dict
        Dictionary which assigns the best transformation for every used prediction model, for every scenario.
    list
        A list of Scenario objects, one for each scenario.
    """
    transformations_to_test = [*param_config["model_parameters"]["possible_transformations"].split(",")]
    main_accuracy_estimator = param_config["model_parameters"]["main_accuracy_estimator"]
    models = [*param_config["model_parameters"]["models"].split(",")]

    best_transformations = dict.fromkeys(models, {})
    scenarios = []

    try:
        max_threads = param_config['max_threads']
    except KeyError:
        try:
            max_threads = len(os.sched_getaffinity(0))
        except:
            max_threads = 1

    columns = ingested_data.columns

    for col in columns:
        model_results = {}
        scenario_data = ingested_data[[col]]
        xcorr = total_xcorr[col] if total_xcorr is not None else None

        for model in models:
            this_model_performances = []

            log.info(f"Using model {model}...")

            for transf in transformations_to_test:
                log.info(f"Computing univariate prediction for {col} using transformation: {transf}...")
                predictor = model_factory(model, param_config=param_config, transformation=transf)
                _result = predictor.launch_model(scenario_data.copy(), max_threads=max_threads)

                performances = _result.results
                performances.sort(key=lambda x: getattr(x.testing_performances, main_accuracy_estimator.upper()))
                performances = getattr(performances[0].testing_performances, main_accuracy_estimator.upper())

                this_model_performances.append((_result, performances, transf))

            this_model_performances.sort(key=lambda x: x[1])
            best_tr = this_model_performances[0][2]
            [log.debug(f"Error with {t}: {e}") for t, e in zip(map(lambda x: x[2], this_model_performances),
                                                               map(lambda x: x[1], this_model_performances))]
            log.info(f"Best transformation for {col} using {model}: {best_tr}")
            best_transformations[model][col] = best_tr
            model_results[model] = this_model_performances[0][0]

        scenarios.append(
            Scenario(scenario_data, model_results, xcorr)
        )

    return best_transformations, scenarios


def get_best_multivariate_predictions(scenarios: [Scenario], ingested_data: DataFrame,
                                      best_transformations: dict, total_xcorr: dict, param_config: dict):
    """
    Starting from the a list of scenarios, use the available univariated predictions to compute new multivariate
    predictions. These new predictions will be used only if better than the univariate ones.

    Returns the updated list of scenarios.

    Parameters
    ----------
    scenarios : [Scenario]
    ingested_data : DataFrame
    best_transformations : dict
    total_xcorr : dict
    param_config : dict

    Returns
    -------
    list
    """
    iterations = 0
    best_forecasts_found = 0

    if total_xcorr is not None:
        xcorr_mode_target = param_config["xcorr_parameters"]["xcorr_mode_target"]
        xcorr_threshold = param_config["xcorr_parameters"]["xcorr_extra_regressor_threshold"]
        main_accuracy_estimator = param_config["model_parameters"]["main_accuracy_estimator"]

    try:
        additional_regressors = param_config["additional_regressors"]
    except KeyError:
        additional_regressors = None

    models = [*param_config["model_parameters"]["models"].split(",")]

    try:
        max_threads = param_config['max_threads']
    except KeyError:
        try:
            max_threads = len(os.sched_getaffinity(0))
        except:
            max_threads = 1

    for model in models:
        log.info(f"Checking optimal predictions with model {model}")
        best_forecasts_found = 0
        iterations = 0

        while best_forecasts_found != len(ingested_data.columns):
            log.info(f"-> Found the optimal prediction for only {best_forecasts_found}")
            best_forecasts_found = 0

            for col in ingested_data.columns:
                useful_extra_regressors = []

                log.debug(f"Look for extra regressors in other dataset's columns...")
                try:
                    local_xcorr = total_xcorr[col][xcorr_mode_target]

                    # Add extra regressors from the original dataset
                    for extra_regressor in local_xcorr.columns:
                        # Look only in correlation with future lags.
                        index_of_max = local_xcorr[extra_regressor].abs().idxmax()
                        corr = local_xcorr.loc[index_of_max, extra_regressor]
                        if abs(corr) > xcorr_threshold and index_of_max >= 0:
                            log.debug(
                                f"Found a possible extra-regressor for {col}: {extra_regressor} at lag {index_of_max}")

                            useful_extra_regressors.append(
                                prepare_extra_regressor(next(filter(
                                    lambda x: x.scenario_data.columns[0] == extra_regressor, scenarios)), model=model))
                    local_xcorr = total_xcorr[col]  # To give the full xcorr to Scenario
                except:
                    local_xcorr = None

                log.debug(f"Look for user-given additional regressors...")
                try:
                    additional_regressor_path = additional_regressors[col]
                    useful_extra_regressors.append(ingest_additional_regressors(additional_regressor_path, param_config))
                except:
                    pass

                if len(useful_extra_regressors) == 0:
                    log.debug(f"No useful extra-regressor found for {col}: skipping...")
                    best_forecasts_found += 1
                else:
                    useful_extra_regressors = reduce(lambda x, y: x.join(y), useful_extra_regressors)
                    log.info(f"Found useful extra-regressors: {useful_extra_regressors.columns}. "
                             f"Re-compute the prediction for {col}")

                    scenario_data = ingested_data[[col]]

                    tr = best_transformations[model][col]

                    predictor = model_factory(model, param_config, transformation=tr)
                    _result = predictor.launch_model(scenario_data.copy(),
                                                     extra_regressors=useful_extra_regressors.copy(),
                                                     max_threads=max_threads)
                    old_this_scenario = next(filter(lambda x: x.scenario_data.columns[0] == col, scenarios))

                    old_errors = [x.testing_performances.MAE for x in old_this_scenario.models[model].results]
                    min_old_error = min(old_errors)
                    min_new_error = min([x.testing_performances.MAE for x in _result.results])

                    if min_new_error < min_old_error:
                        log.info(f"Obtained a better error: {min_new_error} vs old {min_old_error}")
                        new_model_results = old_this_scenario.models
                        new_model_results[model] = _result
                        new_scenario = Scenario(scenario_data, new_model_results, local_xcorr)
                        scenarios = [new_scenario if x.scenario_data.columns[0] == col else x for x in scenarios]
                    else:
                        log.info(f"No improvements.")
                        best_forecasts_found += 1
            iterations += 1

    log.info(f"Found the optimal prediction for all the {best_forecasts_found} scenarios in {iterations} iterations!")
    return scenarios


def get_best_predictions(ingested_data: DataFrame, param_config: dict):

    if "xcorr_parameters" in param_config and len(ingested_data.columns) > 1:
        log.info(f"Computing the cross-correlation...")
        total_xcorr = calc_all_xcorr(ingested_data=ingested_data, param_config=param_config)
    else:
        total_xcorr = None

    best_transformations, scenarios = get_best_univariate_predictions(ingested_data, param_config, total_xcorr)

    if total_xcorr is not None or "additional_regressors" in param_config:
        scenarios = get_best_multivariate_predictions(scenarios=scenarios, ingested_data=ingested_data,
                                                      best_transformations=best_transformations,
                                                      total_xcorr=total_xcorr,
                                                      param_config=param_config)

    return scenarios


def compute_historical_predictions(ingested_data, param_config):
    """

    Parameters
    ----------
    ingested_data
    param_config
    total_xcorr

    Returns
    -------

    """
    input_parameters = param_config["input_parameters"]
    models = [*param_config["model_parameters"]["models"].split(",")]
    save_path = param_config["historical_prediction_parameters"]["save_path"]
    try:
        hist_pred_delta = param_config["historical_prediction_parameters"]["delta"]
    except KeyError:
        hist_pred_delta = 1

    try:
        with open(save_path, 'rb') as file:
            historical_prediction = pickle.load(file)
        log.info(f"Loaded historical prediction from file...")
        current_index = historical_prediction[models[0]].index[-1]
    except FileNotFoundError:
        log.info(f"Historical prediction file not found: computing from the start...")
        starting_index = param_config["historical_prediction_parameters"]["initial_index"]

        if "dateparser_options" in input_parameters:
            dateparser_options = input_parameters["dateparser_options"]
            current_index = dateparser.parse(starting_index, **dateparser_options)
        else:
            current_index = dateparser.parse(starting_index)

        historical_prediction = {}
        for model in models:
            historical_prediction[model] = DataFrame(columns=ingested_data.columns)

    final_index = ingested_data.index[-1]
    log.info(f"Starting index: {current_index}")
    log.info(f"Final index: {final_index}")
    delta_time = 1 * ingested_data.index.freq

    if current_index == final_index:
        log.warning(f"Initial and final index are the same. I am recomputing the last point of historical prediction.")
        current_index = current_index - delta_time * hist_pred_delta

    iterations = 0
    cur = current_index
    fin = final_index

    while cur + delta_time * hist_pred_delta <= fin:
        cur += delta_time * hist_pred_delta
        iterations += 1

    additional_computation = cur != fin
    log.debug(f"Historical computations iterations: {iterations}")
    log.debug(f"Historical additional computation: {additional_computation}")

    for i in range(0, iterations):
        available_data = ingested_data[:current_index]  # Remember: this includes current_index
        log.info(f"Using data from {available_data.index[0]} to {current_index} for training...")

        scenarios = get_best_predictions(available_data, param_config)

        log.info(f"Assigning the historical predictions from {current_index + delta_time} to "
                 f"{current_index + hist_pred_delta * delta_time}")
        for s in scenarios:
            for model in s.models:
                p = s.models[model].best_prediction
                scenario_name = s.scenario_data.columns[0]
                next_preds = p.loc[current_index + delta_time:current_index + hist_pred_delta * delta_time, 'yhat']

                for index, value in next_preds.items():
                    historical_prediction[model].loc[index, scenario_name] = value

        current_index += delta_time * hist_pred_delta

        log.info(f"Saving partial historical prediction to file...")
        with open(save_path, 'wb') as file:
            pickle.dump(historical_prediction, file, protocol=pickle.HIGHEST_PROTOCOL)

    if additional_computation:
        log.info(f"Remaining data less than requested delta time. Computing the best predictions with last data...")
        available_data = ingested_data[:current_index]  # Remember: this includes current_index
        log.info(f"Using data from {available_data.index[0]} to {current_index} for training...")

        scenarios = get_best_predictions(available_data, param_config)

        log.info(f"Assigning the historical predictions from {current_index + delta_time} to "
                 f"{final_index}")
        for s in scenarios:
            for model in s.models:
                p = s.models[model].best_prediction
                scenario_name = s.scenario_data.columns[0]
                next_preds = p.loc[current_index + delta_time:final_index, 'yhat']

                for index, value in next_preds.items():
                    historical_prediction[model].loc[index, scenario_name] = value

        log.info(f"Saving partial historical prediction to file...")
        with open(save_path, 'wb') as file:
            pickle.dump(historical_prediction, file, protocol=pickle.HIGHEST_PROTOCOL)

    available_data = ingested_data
    scenarios = get_best_predictions(available_data, param_config)

    for s in scenarios:
        scenario_name = s.scenario_data.columns[0]
        scenario_hist_predictions = {}
        for model in historical_prediction:
            scenario_hist_predictions[model] = DataFrame(historical_prediction[model].loc[:, scenario_name])
        s.set_historical_prediction(scenario_hist_predictions)

    return scenarios


def create_scenarios(ingested_data: DataFrame, param_config: dict):

    if "historical_prediction_parameters" in param_config:
        log.debug(f"Requested the computation of historical predictions.")
        scenarios = compute_historical_predictions(ingested_data, param_config)
    else:
        if "model_parameters" in param_config:
            log.debug(f"Computing best predictions, without history.")
            scenarios = get_best_predictions(ingested_data, param_config)
        else:
            log.debug(f"Creating scenarios only for data visualization.")
            scenarios = []
            if "xcorr_parameters" in param_config and len(ingested_data.columns) > 1:
                total_xcorr = calc_all_xcorr(ingested_data=ingested_data, param_config=param_config)
            else:
                total_xcorr = None

            for col in ingested_data.columns:
                scenario_data = ingested_data[[col]]
                scenario_xcorr = total_xcorr[col] if total_xcorr is not None else None
                scenarios.append(
                    Scenario(scenario_data, None, scenario_xcorr)
                )

    return scenarios


def model_factory(model_class: str, param_config: dict, transformation: str = None) -> PredictionModel:
    """
    Given the name of the model, return the corresponding PredictionModel.

    Parameters
    ----------
    transformation
    param_config
    model_class : str
        Model type.

    Returns
    -------
    PredictionModel
    """
    if model_class == "fbprophet":
        return FBProphet(params=param_config, transformation=transformation)
    if model_class == "LSTM":
        return LSTM_model(param_config, transformation)
    if model_class == "neuralprophet":
        return NeuralProphetModel(param_config, transformation)
    if model_class == "mockup":
        return MockUpModel(param_config, transformation)
    else:
        return ARIMA(params=param_config, transformation=transformation)
