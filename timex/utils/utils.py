import logging
from functools import reduce

from pandas import DataFrame

from timex.data_prediction.prophet_predictor import FBProphet
from timex.scenario.scenario import Scenario

log = logging.getLogger(__name__)


def prepare_extra_regressor(scenario: Scenario, testing_performance_target: str = 'MAE') -> DataFrame:
    """
    This function receives a Scenario object which includes a prediction for a time-series and indications on the
    prediction errors, along with the entire ingested dataset.
    Then, the best possible prediction (w.r.t a specific indicator, i.e MAE) is taken and appended to the original
    time-series, in order to obtain a DataFrame with the original time series and the best possible prediction.

    The resulting DataFrame is returned.

    Parameters
    ----------
    testing_performance_target : str
    Testing performance indicator to use in order to select the best forecast. Default MAE.

    scenario : Scenario
    Scenario from which an extra-regressor should be extracted.

    Returns
    -------
    df : DataFrame
    DataFrame with the length of the original time-series + prediction lags.
    """
    name = scenario.scenario_data.columns[0]
    model_results = scenario.models[0].results
    model_results.sort(key=lambda x: getattr(x.testing_performances, testing_performance_target.upper()))

    original_ts = scenario.ingested_data[[name]]
    f = model_results[0].prediction.loc[:, ['yhat']]
    f.rename(columns={'yhat': name}, inplace=True)

    best_entire_forecast = original_ts.combine_first(f)

    return best_entire_forecast


def get_best_univariate_predictions(ingested_data: DataFrame, param_config: dict, total_xcorr: dict):
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
    list
        A list of tuples. A tuple is put in the list for every scenario, and it is composed of
        (scenario_name, best_transformation_found). It can be useful in the rest of the program.
    list
        A list of Scenario objects, one for each scenario.
    """
    transformations_to_test = [*param_config["model_parameters"]["possible_transformations"].split(",")]
    main_accuracy_estimator = param_config["model_parameters"]["main_accuracy_estimator"]
    best_transformations = []
    scenarios = []

    columns = ingested_data.columns

    for col in columns:
        scenario_data = ingested_data[[col]]
        model_results = []

        xcorr = total_xcorr[col]

        prophet_results = []

        # TODO: make it usable with every model, not just fbprophet
        for transf in transformations_to_test:
            log.info(f"Computing univariate prediction for {col} using transformation: {transf}...")
            predictor = FBProphet(param_config, transformation=transf)
            prophet_result = predictor.launch_model(scenario_data.copy())

            performances = prophet_result.results
            performances.sort(key=lambda x: getattr(x.testing_performances, main_accuracy_estimator.upper()))
            performances = getattr(performances[0].testing_performances, main_accuracy_estimator.upper())

            prophet_results.append((prophet_result, performances, transf))
            # model_results.append(prophet_result)

            # predictor = ARIMA(param_config)
            # arima_result = predictor.launch_model(scenario_data.copy())
            # model_results.append(arima_result)

        prophet_results.sort(key=lambda x: x[1])
        best_tr = prophet_results[0][2]
        [log.debug(f"Error with {t}: {e}") for t, e in zip(map(lambda x: x[2], prophet_results),
                                                           map(lambda x: x[1], prophet_results))]
        log.info(f"Best transformation for {col}: {best_tr}")
        best_transformations.append((col, best_tr))
        model_results.append(prophet_results[0][0])

        scenarios.append(
            Scenario(scenario_data, model_results, ingested_data, xcorr)
        )

    return best_transformations, scenarios


def get_best_multivariate_predictions(scenarios: [Scenario], ingested_data: DataFrame,
                                      best_transformations: list, total_xcorr: dict, param_config: dict):
    """
    Starting from the a list of scenarios, use the available univariated predictions to compute new multivariate
    predictions. These new predictions will be used only if better than the univariate ones.

    Returns the updated list of scenarios.

    Parameters
    ----------
    scenarios : [Scenario]
    ingested_data : DataFrame
    best_transformations : list
    total_xcorr : dict
    param_config : dict

    Returns
    -------
    list
    """
    iterations = 0
    best_forecasts_found = 0

    xcorr_mode_target = param_config["model_parameters"]["xcorr_mode_target"]
    xcorr_threshold = param_config["model_parameters"]["xcorr_extra_regressor_threshold"]
    main_accuracy_estimator = param_config["model_parameters"]["main_accuracy_estimator"]

    while best_forecasts_found != len(ingested_data.columns):
        log.info(f"-> Found the optimal prediction for only {best_forecasts_found}")
        best_forecasts_found = 0

        for col in ingested_data.columns:
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

                tr = (next(filter(lambda x: x[0] == col, best_transformations)))[1]
                predictor = FBProphet(param_config, transformation=tr)
                prophet_result = predictor.launch_model(scenario_data.copy(),
                                                        extra_regressors=useful_extra_regressors.copy())
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
        iterations += 1

    log.info(f"Found the optimal prediction for all the {best_forecasts_found} scenarios in {iterations} iterations!")
    return scenarios
