import logging
import os
import pickle
from functools import reduce
from typing import Tuple

import dateparser
from pandas import DataFrame

from timexseries.data_ingestion import ingest_additional_regressors
from timexseries.data_prediction import PredictionModel
from timexseries.data_prediction.models.arima_predictor import ARIMAModel
from timexseries.data_prediction.models.exponentialsmoothing_predictor import ExponentialSmoothingModel
from timexseries.data_prediction.models.lstm_predictor import LSTMModel
from timexseries.data_prediction.models.mockup_predictor import MockUpModel
# from timexseries.data_prediction.models.neuralprophet_predictor import NeuralProphetModel
from timexseries.data_prediction.models.prophet_predictor import FBProphetModel
from timexseries.data_prediction.xcorr import calc_all_xcorr
from timexseries.timeseries_container import TimeSeriesContainer

log = logging.getLogger(__name__)


def prepare_extra_regressor(container: TimeSeriesContainer, model: str) -> DataFrame:
    """
    This function receives a `timexseries.timeseries_container.TimeSeriesContainer` object, which includes the time-series
    historical data and the various predictions for the future.

    The best prediction for the model 'model' is taken and appended to the original time-series, in order to obtain a
    DataFrame with the original time series and the best possible prediction.

    The resulting DataFrame is returned.

    Parameters
    ----------
    container : TimeSeriesContainer
        `timexseries.timeseries_container.TimeSeriesContainer` from which an extra-regressor should be extracted.

    model
        The model from which get the best available prediction.

    Returns
    -------
    df : DataFrame
        DataFrame with the length of the original time-series + prediction lags.
    """
    name = container.timeseries_data.columns[0]
    best_prediction = container.models[model].best_prediction

    original_ts = container.timeseries_data
    f = best_prediction.loc[:, ['yhat']]
    f.rename(columns={'yhat': name}, inplace=True)

    best_entire_forecast = original_ts.combine_first(f)

    return best_entire_forecast


def get_best_univariate_predictions(ingested_data: DataFrame, param_config: dict, total_xcorr: dict = None) -> \
        Tuple[dict, list]:
    """
    Compute, for every column in `ingested_data` (every time-series) the best univariate prediction possible.
    This is done using the models specified in `param_config` and testing the effect of the different transformations
    specified in `param_config`. Moreover, the best feature transformation found, across the possible ones, will be
    returned.

    Parameters
    ----------
    ingested_data : DataFrame
        Initial data of the time-series.
    param_config : dict
        TIMEX configuration dictionary. In particular, the `model_parameters` sub-dictionary will be used. In
        `model_parameters` the following options has to be specified:

        - `possible_transformations`: comma-separated list of transformations keywords (e.g. "none,log_modified").
        - `main_accuracy_estimator`: error metric which will be minimized as target by the procedure. E.g. "mae".
        - `models`: comma-separated list of the models to use (e.g. "fbprophet,arima").

    total_xcorr : dict, optional, default None
        Cross-correlation dictionary computed by `calc_all_xcorr`. The cross-correlation is actually not used in this
        function, however it is used to build the returned `timexseries.timeseries_container.TimeSeriesContainer`, if given.

    Returns
    -------
    dict
        Dictionary which assigns the best transformation for every used prediction model, for every time-series.
    list
        A list of `timexseries.timeseries_container.TimeSeriesContainer` objects, one for each time-series.

    Examples
    --------
    Create some fake data:
    >>> dates = pd.date_range('2000-01-01', periods=30)  # Last index is 2000-01-30
    >>> ds = pd.DatetimeIndex(dates, freq="D")
    >>> a = np.arange(30, 60)
    >>> b = np.arange(60, 90)
    >>> timeseries_dataframe = DataFrame(data={"a": a, "b": b}, index=ds)

    And create the model configuration part of the TIMEX configuration dictionary:
    >>> param_config = {
    ...   "model_parameters": {
    ...     "models": "fbprophet",  # Model(s) which will be tested.
    ...     "possible_transformations": "none,log_modified",  # Possible feature transformation to test.
    ...     "main_accuracy_estimator": "mae",
    ...     "delta_training_percentage": 20,  # Training windows will be incremented by the 20% each step...
    ...     "test_values": 5,  # Use the last 5 values as validation set.
    ...     "prediction_lags": 7,  # Predict the next 7 points after 2000-01-30.
    ...     }
    ... }

    Now, get the univariate predictions:
    >>> best_transformations, timeseries_outputs = get_best_univariate_predictions(timeseries_dataframe, param_config)

    Let's inspect the results. `best_transformations` contains the suggested feature transformations to use:
    >>> best_transformations
    {'fbprophet': {'a': 'none', 'b': 'none'}}

    It is reasonable with this simple data that no transformation is the best transformation.
    We have the `timexseries.timeseries_container.TimeSeriesContainer` list as well:
    >>> timeseries_outputs
    [<timexseries.timeseries_container.TimeSeriesContainer at 0x7f62f45d1fa0>,
     <timexseries.timeseries_container.TimeSeriesContainer at 0x7f62d4e97cd0>]

    These are the `timexseries.timeseries_container.TimeSeriesContainer` objects, one for time-series `a` and one for `b`.
    Each one has various fields, in this case the most interesting one is `models`:
    >>> timeseries_outputs[0].models
    {'fbprophet': <timexseries.data_prediction.models.predictor.ModelResult at 0x7f62f45d1d90>}

    This is the `timexseries.data_prediction.models.predictor.ModelResult` object for FBProphet that we have just computed.
    """
    transformations_to_test = [*param_config["model_parameters"]["possible_transformations"].split(",")]
    main_accuracy_estimator = param_config["model_parameters"]["main_accuracy_estimator"]
    models = [*param_config["model_parameters"]["models"].split(",")]

    best_transformations = dict.fromkeys(models, {})
    timeseries_containers = []

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
        timeseries_data = ingested_data[[col]]
        xcorr = total_xcorr[col] if total_xcorr is not None else None

        for model in models:
            this_model_performances = []

            log.info(f"Using model {model}...")

            for transf in transformations_to_test:
                log.info(f"Computing univariate prediction for {col} using transformation: {transf}...")
                predictor = model_factory(model, param_config=param_config, transformation=transf)
                _result = predictor.launch_model(timeseries_data.copy(), max_threads=max_threads)

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

        timeseries_containers.append(
            TimeSeriesContainer(timeseries_data, model_results, xcorr)
        )

    return best_transformations, timeseries_containers


def get_best_multivariate_predictions(timeseries_containers: [TimeSeriesContainer], ingested_data: DataFrame,
                                      best_transformations: dict, total_xcorr: dict, param_config: dict):
    """
    Starting from the a list of `timexseries.timeseries_container.TimeSeriesContainer`, use the available univariated
    predictions and the time-series in `ingested_data`, plus eventual user-given additional regressors to compute new
    multivariate predictions.

    These new predictions will be used only if better than the univariate ones.

    Returns the updated list of `timexseries.timeseries_container.TimeSeriesContainer`.

    Parameters
    ----------
    timeseries_containers : [TimeSeriesContainer]
        Initial `timexseries.timeseries_container.TimeSeriesContainer` list from which the computation of multivariate
        predictions start. Some univariate predictions should be already present in each object: more formally, each
        `timexseries.timeseries_container.TimeSeriesContainer` should have the `model_results` attribute.
    ingested_data : DataFrame
        Initial data of the time-series.
    best_transformations : dict
        Dictionary which assigns the best transformation for every used prediction model, for every time-series. It
        should be returned by `get_best_univariate_predictions`.
    total_xcorr : dict
        Cross-correlation dictionary computed by `timexseries.data_prediction.xcorr.calc_all_xcorr`. The cross-correlation is
        used in this function, to find, among all the time-series in `ingested_data`, additional regressors for each
        time-series, if there are some.
    param_config : dict
        TIMEX configuration dictionary. In particular, the `xcorr_parameters` sub-dictionary will be used. In
        `xcorr_parameters` the following options has to be specified if `total_xcorr` parameter is not None:

        - `xcorr_mode_target`: which cross-correlation algorithm should be used as target in evaluating useful
          additional regressors. E.g. "pearson".
        - `xcorr_extra_regressor_threshold`: the minimum absolute value of cross-correlation which indicates a useful
          extra-regressor. E.g. 0.8.

        Additionally, the `additional_regressors` part of the TIMEX configuration parameter dictionary can be used by
        the user to specify additional CSV paths to time-series data to use as extra-regressor.
        It should be a dictionary in the form "target time-series": "path of the additional extra-regressors".
        The key "_all" is a special key which indicates a path to additional extra-regressors which will be used for
        any time-series.

    Returns
    -------
    list
        A list of `timexseries.timeseries_container.TimeSeriesContainer` objects, one for each time-series.

    Examples
    --------
    We will create ad-hoc time-series in which using a multivariate model will perform better than using a univariate
    one.

    >>> dates = pd.date_range('2000-01-01', periods=30)  # Last index is 2000-01-30
    >>> ds = pd.DatetimeIndex(dates, freq="D")
    >>>
    >>> x = np.linspace(0, 2 * np.pi, 60)
    >>> y = np.sin(x)
    >>>
    >>> np.random.seed(0)
    >>> noise = np.random.normal(0, 2.0, 60)
    >>> y = y + noise
    >>>
    >>> a = y[:30]
    >>> b = y[5:35]
    >>>
    >>> timeseries_dataframe = DataFrame(data={"a": a, "b": b}, index=ds)

    In this dataset the time-series `b` can be used to better predict `a`... simply because it is the same series, but
    traslated!

    Try to perform the computations.

    >>> param_config = {
    ...     "model_parameters": {
    ...         "models": "LSTM",
    ...         "possible_transformations": "none,log_modified",
    ...         "main_accuracy_estimator": "mae",
    ...         "delta_training_percentage": 20,
    ...         "test_values": 5,
    ...         "prediction_lags": 7,
    ...     },
    ...     "xcorr_parameters": {
    ...         "xcorr_mode_target": "pearson",
    ...         "xcorr_extra_regressor_threshold": 0.7,
    ...         "xcorr_max_lags": 6,
    ...         "xcorr_mode": "pearson"
    ...     }
    ... }
    >>> xcorr = calc_all_xcorr(timeseries_dataframe, param_config)
    >>> best_transformations, timeseries_outputs = get_best_univariate_predictions(timeseries_dataframe, param_config)
    >>> timeseries_outputs = get_best_multivariate_predictions(timeseries_outputs, timeseries_dataframe,
    >>>                                                        best_transformations, xcorr, param_config)

    From the log, we can see:
    >>> "INFO:timexseries.data_prediction.pipeline:Found useful extra-regressors: Index(['b'], dtype='object'). Re-compute the prediction for a"
    >>> "INFO:timexseries.data_prediction.models.predictor:Creating a LSTM model..."
    >>> "INFO:timexseries.data_prediction.models.predictor:Model will use 5 different training sets..."
    >>> "INFO:timexseries.data_prediction.models.predictor:LSTM/NeuralProphet model. Cant use multiprocessing."
    >>> "INFO:timexseries.data_prediction.pipeline:Obtained a better error: 1.6009327718008979 vs old 1.9918351002921089"

    This means that using `b` as additional regressor for `a` made us obtain a better error.
    """
    iterations = 0
    best_forecasts_found = 0

    if total_xcorr is not None:
        xcorr_mode_target = param_config["xcorr_parameters"]["xcorr_mode_target"]
        xcorr_threshold = param_config["xcorr_parameters"]["xcorr_extra_regressor_threshold"]

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
                depends_on_other_ts = False
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
                                    lambda x: x.timeseries_data.columns[0] == extra_regressor, timeseries_containers)), model=model))
                            depends_on_other_ts = True
                    local_xcorr = total_xcorr[col]  # To give the full xcorr to Scenario
                except:
                    local_xcorr = None

                log.debug(f"Look for user-given additional regressors...")
                try:
                    additional_regressor_path = additional_regressors["_all"]
                    useful_extra_regressors.append(ingest_additional_regressors(additional_regressor_path, param_config))
                except:
                    pass

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

                    timeseries_data = ingested_data[[col]]

                    tr = best_transformations[model][col]

                    predictor = model_factory(model, param_config, transformation=tr)
                    _result = predictor.launch_model(timeseries_data.copy(),
                                                     extra_regressors=useful_extra_regressors.copy(),
                                                     max_threads=max_threads)
                    old_this_container = next(filter(lambda x: x.timeseries_data.columns[0] == col, timeseries_containers))

                    old_errors = [x.testing_performances.MAE for x in old_this_container.models[model].results]
                    min_old_error = min(old_errors)
                    min_new_error = min([x.testing_performances.MAE for x in _result.results])

                    if min_new_error < min_old_error:
                        log.info(f"Obtained a better error: {min_new_error} vs old {min_old_error}")
                        new_model_results = old_this_container.models
                        new_model_results[model] = _result
                        new_container = TimeSeriesContainer(timeseries_data, new_model_results, local_xcorr)
                        timeseries_containers = [new_container if x.timeseries_data.columns[0] == col else x for x in timeseries_containers]
                        if not depends_on_other_ts:
                            best_forecasts_found += 1
                    else:
                        log.info(f"No improvements.")
                        best_forecasts_found += 1
            iterations += 1

    log.info(f"Found the optimal prediction for all the {best_forecasts_found} time-series in {iterations} iterations!")
    return timeseries_containers


def get_best_predictions(ingested_data: DataFrame, param_config: dict):
    """
    Starting from `ingested_data`, using the models/cross correlation settings set in `param_config`, return the best
    possible predictions in a `timexseries.timeseries_container.TimeSeriesContainer` for each time-series in `ingested_data`.

    Parameters
    ----------
    ingested_data : DataFrame
        Initial data of the time-series.

    param_config : dict
        TIMEX configuration dictionary. `get_best_univariate_predictions` and `get_best_multivariate_predictions` will
        use the various settings in `param_config`.

    Returns
    -------
    list
        A list of `timexseries.timeseries_container.TimeSeriesContainer` objects, one for each time-series.

    Examples
    --------
    This is basically the function on top of `get_best_univariate_predictions` and `get_best_multivariate_predictions`:
    it will call first the univariate and then the multivariate if the cross-correlation section is present
    in `param_config`.

    Create some data:
    >>> dates = pd.date_range('2000-01-01', periods=30)  # Last index is 2000-01-30
    >>> ds = pd.DatetimeIndex(dates, freq="D")
    >>> a = np.arange(30, 60)
    >>> b = np.arange(60, 90)
    >>>
    >>> timeseries_dataframe = DataFrame(data={"a": a, "b": b}, index=ds)

    Simply compute the predictions and get the returned `timexseries.timeseries_container.TimeSeriesContainer` objects:
    >>> timeseries_outputs = get_best_predictions(timeseries_dataframe, param_config)
    """
    if "xcorr_parameters" in param_config and len(ingested_data.columns) > 1:
        log.info(f"Computing the cross-correlation...")
        total_xcorr = calc_all_xcorr(ingested_data=ingested_data, param_config=param_config)
    else:
        total_xcorr = None

    best_transformations, timeseries_containers = get_best_univariate_predictions(ingested_data, param_config,
                                                                                  total_xcorr)

    if total_xcorr is not None or "additional_regressors" in param_config:
        timeseries_containers = get_best_multivariate_predictions(timeseries_containers=timeseries_containers, ingested_data=ingested_data,
                                                      best_transformations=best_transformations,
                                                      total_xcorr=total_xcorr,
                                                      param_config=param_config)

    return timeseries_containers


def compute_historical_predictions(ingested_data, param_config):
    """
    Compute the historical predictions, i.e. the predictions for (part) of the history of the time-series.

    Parameters
    ----------
    ingested_data : DataFrame
        Initial data of the time-series.
    param_config : dict
        TIMEX configuration dictionary. In particular, the `historical_prediction_parameters` sub-dictionary will be
        used. In `historical_prediction_parameters` the following options has to be specified:

        - `initial_index`: the point from which the historical computations will be made;
        - `save_path`: the historical computations are saved on a file, serialized with pickle. This allows the re-use
        of these predictions if TIMEX is restarted in the future.

        Additionally, the parameter `delta` can be specified: this indicates how many data points should be predicted
        every run. The default is `1`; a number greater than `1` will reduce the accuracy of the predictions because
        multiple points are predicted with the same model, but it will speed up the computation.

        `input_parameters` will be used because the `initial_index` date will be parsed with the same format provided in
        `input_parameters`, if any. Otherwise the standard `yyyy-mm-dd` format will be used.

    Returns
    -------
    list
        A list of `timexseries.timeseries_container.TimeSeriesContainer` objects, one for each time-series. These containers
        have the `historical_prediction` attribute; the predictions in `model_results` are the more recent available
        ones.

    Notes
    -----
    Historical predictions are predictions computed on past points of the time-series, but using only the data available
    until that point.

    Consider a time-series with length `p`. Consider that we want to find the historical predictions starting from the
    middle of the time-series, i.e. from index `s=p/2`.

    To do that, we take the data available from the start of the time-series to `s`, compute the prediction for the
    instant `s + 1`, and then move forward.

    When all the data has been used, we have `s` predictions, but also the real data corresponding to that predictions;
    this allows the user to check error metrics and understand the real performances of a model, on data never seen.

    These metrics can give an idea of the future performance of the model.

    Examples
    --------
    Create some sample data.
    >>> dates = pd.date_range('2000-01-01', periods=30)  # Last index is 2000-01-30
    >>> ds = pd.DatetimeIndex(dates, freq="D")
    >>> a = np.arange(30, 60)
    >>> b = np.arange(60, 90)
    >>>
    >>> timeseries_dataframe = DataFrame(data={"a": a, "b": b}, index=ds)

    Create the configuration parameters dictionary:
    >>> param_config = {
    ...     "input_parameters": {
    ...     },
    ...     "model_parameters": {
    ...         "models": "fbprophet",
    ...         "possible_transformations": "none,log_modified",
    ...         "main_accuracy_estimator": "mae",
    ...         "delta_training_percentage": 20,
    ...         "test_values": 5,
    ...         "prediction_lags": 7,
    ...     },
    ...     "historical_prediction_parameters": {
    ...         "initial_index": "2000-01-25",
    ...         "save_path": "example.pkl"
    ...     }
    ... }

    Launch the computation.
    >>> timeseries_outputs = compute_historical_predictions(timeseries_dataframe, param_config)

    Similarly to `get_best_predictions`, we have a list of `timexseries.timeseries_container.TimeSeriesContainer` objects.
    However, these objects also have an historical prediction:
    >>> timeseries_outputs[0].historical_prediction
    {'fbprophet':                  a
                 2000-01-26       55
                 2000-01-27       56
                 2000-01-28  56.3552
                 2000-01-29  58.1709
                 2000-01-30  58.9167
    }

    If multiple models were specified, `historical_prediction` dictionary would have other entries.
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

        timeseries_containers = get_best_predictions(available_data, param_config)

        log.info(f"Assigning the historical predictions from {current_index + delta_time} to "
                 f"{current_index + hist_pred_delta * delta_time}")
        for s in timeseries_containers:
            for model in s.models:
                p = s.models[model].best_prediction
                timeseries_name = s.timeseries_data.columns[0]
                next_preds = p.loc[current_index + delta_time:current_index + hist_pred_delta * delta_time, 'yhat']

                for index, value in next_preds.items():
                    historical_prediction[model].loc[index, timeseries_name] = value

        current_index += delta_time * hist_pred_delta

        log.info(f"Saving partial historical prediction to file...")
        with open(save_path, 'wb') as file:
            pickle.dump(historical_prediction, file, protocol=pickle.HIGHEST_PROTOCOL)

    if additional_computation:
        log.info(f"Remaining data less than requested delta time. Computing the best predictions with last data...")
        available_data = ingested_data[:current_index]  # Remember: this includes current_index
        log.info(f"Using data from {available_data.index[0]} to {current_index} for training...")

        timeseries_containers = get_best_predictions(available_data, param_config)

        log.info(f"Assigning the historical predictions from {current_index + delta_time} to "
                 f"{final_index}")
        for s in timeseries_containers:
            for model in s.models:
                p = s.models[model].best_prediction
                timeseries_name = s.timeseries_data.columns[0]
                next_preds = p.loc[current_index + delta_time:final_index, 'yhat']

                for index, value in next_preds.items():
                    historical_prediction[model].loc[index, timeseries_name] = value

        log.info(f"Saving partial historical prediction to file...")
        with open(save_path, 'wb') as file:
            pickle.dump(historical_prediction, file, protocol=pickle.HIGHEST_PROTOCOL)

    available_data = ingested_data
    timeseries_containers = get_best_predictions(available_data, param_config)

    for s in timeseries_containers:
        timeseries_name = s.timeseries_data.columns[0]
        timeseries_historical_predictions = {}
        for model in historical_prediction:
            timeseries_historical_predictions[model] = DataFrame(historical_prediction[model].loc[:, timeseries_name])
        s.set_historical_prediction(timeseries_historical_predictions)

    return timeseries_containers


def create_timeseries_containers(ingested_data: DataFrame, param_config: dict):
    """
    Entry points of the pipeline; it will compute univariate/multivariate predictions, historical predictions, or only
    create the containers with the time-series data, according to the content of `param_config`, with this logic:

    - if `historical_prediction_parameters` is in `param_config`, then `compute_historical_predictions` will be called;
    - else, if `model_parameters` is in `param_config`, then `get_best_predictions` will be called;
    - else, create a list of `timexseries.timeseries_container.TimeSeriesContainer` with only the time-series data and, if
      `xcorr_parameters` is in `param_config`, with also the cross-correlation.

    Parameters
    ----------
    ingested_data : DataFrame
        Initial data of the time-series.

    param_config : dict
        TIMEX configuration dictionary.

    Returns
    -------
    list
        A list of `timexseries.timeseries_container.TimeSeriesContainer` objects, one for each time-series.

    Examples
    --------
    The first example of `compute_historical_predictions` applies also here; calling `create_timeseries_containers` will
    produce the same identical result.

    If we remove `historical_prediction_parameters` from the `param_config`, then calling this function is the same as
    calling `get_best_predictions`.

    However, if no predictions should be made but we just want the time-series containers:
    >>> dates = pd.date_range('2000-01-01', periods=30)  # Last index is 2000-01-30
    >>> ds = pd.DatetimeIndex(dates, freq="D")
    >>> a = np.arange(30, 60)
    >>> b = np.arange(60, 90)
    >>> timeseries_dataframe = DataFrame(data={"a": a, "b": b}, index=ds)

    Create the containers:
    >>> param_config = {}
    >>> timeseries_outputs = create_timeseries_containers(timeseries_dataframe, param_config)

    Check that no models, no historical predictions and no cross-correlation are present in the containers:
    >>> print(timeseries_outputs[0].models)
    None
    >>> print(timeseries_outputs[0].historical_prediction)
    None
    >>> print(timeseries_outputs[0].xcorr)
    None

    If `xcorr_parameters` was specified, then the last command would not return None.
    Check that the time-series data is there:

    >>> print(timeseries_outputs[0].timeseries_data)
                 a
    2000-01-01  30
    2000-01-02  31
    ...
    2000-01-29  58
    2000-01-30  59
    """
    if "historical_prediction_parameters" in param_config:
        log.debug(f"Requested the computation of historical predictions.")
        timeseries_containers = compute_historical_predictions(ingested_data, param_config)
    else:
        if "model_parameters" in param_config:
            log.debug(f"Computing best predictions, without history.")
            timeseries_containers = get_best_predictions(ingested_data, param_config)
        else:
            log.debug(f"Creating containers only for data visualization.")
            timeseries_containers = []
            if "xcorr_parameters" in param_config and len(ingested_data.columns) > 1:
                total_xcorr = calc_all_xcorr(ingested_data=ingested_data, param_config=param_config)
            else:
                total_xcorr = None

            for col in ingested_data.columns:
                timeseries_data = ingested_data[[col]]
                timeseries_xcorr = total_xcorr[col] if total_xcorr is not None else None
                timeseries_containers.append(
                    TimeSeriesContainer(timeseries_data, None, timeseries_xcorr)
                )

    return timeseries_containers


def model_factory(model_class: str, param_config: dict, transformation: str = None) -> PredictionModel:
    """
    Given the name of the model, return the corresponding PredictionModel.

    Parameters
    ----------
    model_class : str
        Model type, e.g. "fbprophet"
    param_config : dict
        TIMEX configuration dictionary, to pass to the just created model.
    transformation : str, optional, default None
        Optional `transformation` parameter to pass to the just created model.

    Returns
    -------
    PredictionModel
        Prediction model of the class specified in `model_class`.

    Examples
    --------
    >>> param_config = {
    ...    "model_parameters": {
    ...        "possible_transformations": "none,log_modified",
    ...        "main_accuracy_estimator": "mae",
    ...        "delta_training_percentage": 20,
    ...        "test_values": 5,
    ...        "prediction_lags": 7,
    ...    },
    ...}

    >>> model = model_factory("fbprophet", param_config, "none")
    >>> print(type(model))
    <class 'timexseries.data_prediction.models.prophet_predictor.FBProphetModel'>
    """
    model_class = model_class.lower()
    if model_class == "fbprophet":
        return FBProphetModel(params=param_config, transformation=transformation)
    if model_class == "lstm":
        return LSTMModel(param_config, transformation)
    # if model_class == "neuralprophet":
    #     return NeuralProphetModel(param_config, transformation)
    if model_class == "mockup":
        return MockUpModel(param_config, transformation)
    if model_class == "exponentialsmoothing":
        return ExponentialSmoothingModel(param_config, transformation)
    else:
        return ARIMAModel(params=param_config, transformation=transformation)