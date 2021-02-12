import json
import logging
import math
import multiprocessing
import pkgutil
from functools import reduce

import pandas as pd
from pandas import DataFrame

from timex.data_prediction.transformation import transformation_factory
from timex.data_prediction.validation_performances import ValidationPerformance

log = logging.getLogger(__name__)


class SingleResult:
    """
    Class for the result of a model trained on a specific training set.

    Attributes
    ----------
    prediction : DataFrame
        Estimated prediction, using this training set
    testing_performances : timex.data_prediction.validation_performances.ValidationPerformance
        Testing performance, on the test set, obtained using
        this training set to train the model.
    """

    def __init__(self, prediction: DataFrame, testing_performances: ValidationPerformance):
        self.prediction = prediction
        self.testing_performances = testing_performances


class ModelResult:
    """
    Class for the result of a model trained on a time series.
    This will include predicted data, performance indexes and model
    parameters, able to fully characterize the model itself.

    Attributes
    ----------
    results : [SingleResult]
        List of all the result obtained using all the possible training set
        for this model, on the time series.
    characteristics : dict
        Model parameters, obtained by automatic tuning.
    best_prediction : DataFrame
        Prediction obtained using the best training window and _all_ the available points in the time series.
    """

    def __init__(self, results: [SingleResult], characteristics: dict, best_prediction: DataFrame):
        self.results = results
        self.characteristics = characteristics
        self.best_prediction = best_prediction


class PredictionModel:
    """
    Base class for every prediction model which can be used on a time series.

    Attributes
    ----------
    freq : str
        If available, the frequency of the time series. Default None
    test_percentage : float
        Percentage of the time series to be used for the test set. Default 0
    test_values : int
        Absolute number of observations used for the test set. Default 0
    transformation : str
        Transformation to apply to the time series before using it. Default None
    prediction_lags : int
        Number of future lags for which the prediction has to be made. Default 0
    model_characteristics : dict
        Dictionary of values containing the main characteristics and parameters
        of the model. Default {}
    """

    def __init__(self, params: dict, name: str, transformation: str = None) -> None:
        self.name = name

        log.info(f"Creating a {self.name} model...")

        if "model_parameters" not in params:
            log.debug(f"Loading default settings...")
            parsed = pkgutil.get_data(__name__, "default_prediction_parameters/" + self.name + ".json")
            model_parameters = json.loads(parsed)
        else:
            log.debug(f"Loading user settings...")
            model_parameters = params["model_parameters"]

        if "test_values" in model_parameters:
            self.test_values = model_parameters["test_values"]
        else:
            self.test_percentage = model_parameters["test_percentage"]
            self.test_values = -1

        if transformation is not None:
            self.transformation = transformation_factory(transformation)
        else:
            self.transformation = transformation_factory(model_parameters["transformation"])

        self.prediction_lags = model_parameters["prediction_lags"]
        self.delta_training_percentage = model_parameters["delta_training_percentage"]
        self.main_accuracy_estimator = model_parameters["main_accuracy_estimator"]
        self.delta_training_values = 0
        self.model_characteristics = {}

        self.freq = ""
        log.debug(f"Finished model creation.")

    def train(self, ingested_data: DataFrame, extra_regressor: DataFrame = None):
        """
        Train the model on the ingested_data.

        Parameters
        ----------
        ingested_data : DataFrame
            DataFrame which _HAS_ to be divided in training and test set.

        extra_regressor : DataFrame
            Additional time series to use for better predictions.
        """
        pass

    def predict(self, future_dataframe: DataFrame, extra_regressor: DataFrame = None) -> DataFrame:
        """
        Return a DataFrame with the shape of future_dataframe,  filled with predicted values.

        Returns
        -------
        forecast : DataFrame
        """
        pass

    def compute_trainings(self, train_ts: DataFrame, test_ts: DataFrame, extra_regressors: DataFrame, max_threads: int):
        """
        Compute the training of a model on a set of different training sets, of increasing length.
        train_ts is split in train_sets_number and the computation is split across different processes, according to the
        value of max_threads which indicates the maximum number of usable processes.

        Parameters
        ----------
        train_ts : DataFrame
            The entire train_ts which can be used; it will be split in different training sets, in order to test which
            length performs better.
        test_ts : DataFrame
            Testing set to be used to compute the models' performances.
        extra_regressors : DataFrame
            Additional values to be passed to train_model in order to improve the performances.
        max_threads : int
            Maximum number of threads to use in the training phase.

        Returns
        -------
        results : list
            List of SingleResult. Each one is the result relative to the use of a specific train set.
        """
        train_sets_number = math.ceil(len(train_ts) / self.delta_training_values)
        log.info(f"Model will use {train_sets_number} different training sets...")

        def c(targets: [int], _return_dict: dict, thread_number: int):
            _results = []

            for _i in range(targets[0], targets[1]):
                tr = train_ts.iloc[-(_i+1) * self.delta_training_values:]

                log.debug(f"Trying with last {len(tr)} values as training set, in thread {thread_number}")

                self.train(tr.copy(), extra_regressors)

                future_df = pd.DataFrame(index=pd.date_range(freq=self.freq,
                                                             start=tr.index.values[0],
                                                             periods=len(tr) + self.test_values + self.prediction_lags),
                                         columns=["yhat"], dtype=tr.iloc[:, 0].dtype)

                forecast = self.predict(future_df, extra_regressors)

                forecast.loc[:, 'yhat'] = self.transformation.inverse(forecast['yhat'])
                try:
                    forecast.loc[:, 'yhat_lower'] = self.transformation.inverse(forecast['yhat_lower'])
                    forecast.loc[:, 'yhat_upper'] = self.transformation.inverse(forecast['yhat_upper'])
                except:
                    pass

                testing_prediction = forecast.iloc[-self.prediction_lags - self.test_values:-self.prediction_lags]

                first_used_index = tr.index.values[0]

                tp = ValidationPerformance(first_used_index)
                tp.set_testing_stats(test_ts.iloc[:, 0], testing_prediction["yhat"])
                _results.append(SingleResult(forecast, tp))

            _return_dict[thread_number] = _results

        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        processes = []

        if self.name == 'LSTM' or self.name == 'NeuralProphet':
            log.info(f"LSTM/NeuralProphet model. Cant use multiprocessing.")
            return_d = {}
            distributions = [[0, train_sets_number]]
            c(distributions[0], return_d, 0)
            return return_d[0]

        if max_threads == 1:
            distributions = [[0, train_sets_number]]
            n_threads = 1
        else:
            distributions = []

            if train_sets_number % max_threads == 0:
                n_threads = max_threads
                subtraining_dim = train_sets_number // n_threads
                for i in range(0, n_threads):
                    distributions.append([i*subtraining_dim, i*subtraining_dim + subtraining_dim])
            else:
                n_threads = min(max_threads, train_sets_number)
                subtraining_dim = train_sets_number // n_threads
                for i in range(0, n_threads):
                    distributions.append([i*subtraining_dim, i*subtraining_dim+subtraining_dim])
                for k in range(0, (train_sets_number % n_threads)):
                    distributions[k][1] += 1
                    distributions[k+1::] = [ [x+1, y+1] for x, y in distributions[k+1::]]

        for i in range(0, n_threads):
            processes.append(multiprocessing.Process(target=c, args=(distributions[i], return_dict, i)))
            processes[-1].start()

        for p in processes:
            p.join()

        results = reduce(lambda x, y: x+y, [return_dict[key] for key in return_dict])

        return results

    def compute_best_prediction(self, ingested_data: DataFrame, training_results: [SingleResult], extra_regressors = None):
        """
        Given the ingested data and the training results, identify the best training window and compute a prediction
        using all the possible data, till the end of the series.
        Parameters
        ----------
        extra_regressors
        ingested_data
        training_results

        Returns
        -------
        DataFrame
            Best found prediction
        """
        training_results.sort(key=lambda x: getattr(x.testing_performances, self.main_accuracy_estimator.upper()))
        best_starting_index = training_results[0].testing_performances.first_used_index

        training_data = ingested_data.copy().loc[best_starting_index:]

        training_data.iloc[:, 0] = self.transformation.apply(training_data.iloc[:, 0])

        self.train(training_data.copy(), extra_regressors)

        future_df = pd.DataFrame(index=pd.date_range(freq=self.freq,
                                                     start=training_data.index.values[0],
                                                     periods=len(training_data) + self.prediction_lags),
                                 columns=["yhat"], dtype=training_data.iloc[:, 0].dtype)

        forecast = self.predict(future_df, extra_regressors)
        forecast.loc[:, 'yhat'] = self.transformation.inverse(forecast['yhat'])

        try:
            forecast.loc[:, 'yhat_lower'] = self.transformation.inverse(forecast['yhat_lower'])
            forecast.loc[:, 'yhat_upper'] = self.transformation.inverse(forecast['yhat_upper'])
        except:
            pass

        return forecast

    def launch_model(self, ingested_data: DataFrame, extra_regressors: DataFrame = None, max_threads: int = 1) -> ModelResult:
        """
        Train the model on ingested_data and returns a ModelResult object.

        Parameters
        ----------
        ingested_data : DataFrame
            DataFrame containing the historical time series value; it will be split in training and test parts.
        extra_regressors : DataFrame
            Additional values to be passed to train_model in order to improve the performances.
        max_threads : int
            Maximum number of threads to use in the training phase.

        Returns
        -------
        model_result : ModelResult
            Object containing the results of the model, trained on ingested_data.
        """
        model_characteristics = self.model_characteristics

        self.delta_training_values = int(round(len(ingested_data) * self.delta_training_percentage / 100))

        if self.test_values == -1:
            self.test_values = int(round(len(ingested_data) * (self.test_percentage / 100)))

        self.freq = pd.infer_freq(ingested_data.index)

        # We need to pass ingested data both to compute_training and compute_best_prediction, so better use copy()
        # because, otherwise, we may have side effects.
        train_ts = ingested_data.copy().iloc[:-self.test_values]
        test_ts = ingested_data.copy().iloc[-self.test_values:]

        train_ts.iloc[:, 0] = self.transformation.apply(train_ts.iloc[:, 0])

        model_training_results = self.compute_trainings(train_ts, test_ts, extra_regressors, max_threads)

        best_prediction = self.compute_best_prediction(ingested_data, model_training_results, extra_regressors)

        if extra_regressors is not None:
            model_characteristics["extra_regressors"] = ', '.join([*extra_regressors.columns])

        model_characteristics["name"] = self.name
        model_characteristics["delta_training_percentage"] = self.delta_training_percentage
        model_characteristics["delta_training_values"] = self.delta_training_values
        model_characteristics["test_values"] = self.test_values
        model_characteristics["transformation"] = self.transformation

        return ModelResult(results=model_training_results, characteristics=model_characteristics,
                           best_prediction=best_prediction)


