import json
import logging
import math
import multiprocessing
import pkgutil
from functools import reduce
from math import sqrt

import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from timex.data_prediction.transformation import transformation_factory

log = logging.getLogger(__name__)


class TestingPerformance:
    """
    Class for the summary of various statistical indexes relative
    to the performance of a prediction model.

    Attributes
    ----------
    first_used_index :
        Index of the first value used in the time series to generate
        these results.
    MSE : float
        Mean Squared Error. Default 0
    RMSE: float
        Root Mean Squared Error. Default 0
    MAE: float
        Mean Absolute Error. Default 0
    AM: float
        Arithmetic Mean of error. Default 0
    """

    def __init__(self, first_used_index):
        self.first_used_index = first_used_index
        self.MSE = 0
        self.RMSE = 0
        self.MAE = 0
        self.AM = 0

    def set_testing_stats(self, actual: DataFrame, predicted: DataFrame):
        """
        Set all the statistical indexes according do input data.

        Parameters
        ----------
        actual : DataFrame
            Actual data stored in a DataFrame.
        predicted : DataFrame
            Data predicted by a model, stored in a DataFrame.
        """
        self.MSE = mean_squared_error(actual, predicted)
        self.MAE = mean_absolute_error(actual, predicted)
        self.RMSE = sqrt(self.MSE)
        self.AM = sum([y - yhat for y, yhat in zip(actual, predicted)]) / len(actual)

    def get_dict(self) -> dict:
        """
        Return all the parameters, in a dict.

        Returns
        d : dict
            All the statistics, in a dict.
        """
        d = {}
        for attribute, value in self.__dict__.items():
            d[attribute] = value

        return d


class SingleResult:
    """
    Class for the result of a model trained on a specific training set.

    Attributes
    ----------
    prediction : DataFrame
        Estimated prediction, using this training set
    testing_performances : TestingPerformance
        Testing performance, on the test set, obtained using
        this training set to train the model.
    """

    def __init__(self, prediction: DataFrame, testing_performances: TestingPerformance):
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
    """

    def __init__(self, results: [SingleResult], characteristics: dict):
        self.results = results
        self.characteristics = characteristics


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

    def train(self, ingested_data: DataFrame, extra_regressor: DataFrame = None) -> TestingPerformance:
        """
        Train the model on the ingested_data.
        Returns the statistical performance of the testing.

        Parameters
        ----------
        ingested_data : DataFrame
            DataFrame which _HAS_ to be divided in training and test set.

        extra_regressor : DataFrame
            Additional time series to use for better predictions.

        Returns
        -------
        testing_performance : TestingPerformance
        """
        pass

    def predict(self, future_dataframe: DataFrame, extra_regressor: DataFrame = None) -> DataFrame:
        """
        Return a DataFrame with the shape of future_dataframe,
        filled with predicted values.

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
                testing_prediction = forecast.iloc[-self.prediction_lags - self.test_values:-self.prediction_lags]

                first_used_index = tr.index.values[0]

                tp = TestingPerformance(first_used_index)
                tp.set_testing_stats(test_ts.iloc[:, 0], testing_prediction["yhat"])
                _results.append(SingleResult(forecast, tp))

            _return_dict[thread_number] = _results

        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        processes = []

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

        train_ts = ingested_data.iloc[:-self.test_values]
        test_ts = ingested_data.iloc[-self.test_values:]

        with pd.option_context('mode.chained_assignment', None):
            train_ts.iloc[:, 0] = self.transformation.apply(train_ts.iloc[:, 0])

        results = self.compute_trainings(train_ts, test_ts, extra_regressors, max_threads)

        if extra_regressors is not None:
            model_characteristics["extra_regressors"] = ', '.join([*extra_regressors.columns])

        model_characteristics["name"] = self.name
        model_characteristics["delta_training_percentage"] = self.delta_training_percentage
        model_characteristics["delta_training_values"] = self.delta_training_values
        model_characteristics["test_values"] = self.test_values
        model_characteristics["transformation"] = self.transformation

        return ModelResult(results=results, characteristics=model_characteristics)


# def pre_transformation(data: Series, transformation: str) -> Series:
#     """
#     Applies a function (whose name is defined in transformation) to the input data.
#     Returns the transformed data.
#
#     Parameters
#     ----------
#     data : Series
#         Pandas Series. Transformation will be applied to each value.
#     transformation : str
#         Name of the transformation which should be applied.
#
#     Returns
#     -------
#     transformed_data : Series
#         Series where the transformation has been applied.
#     """
#     if transformation == "log":
#         # def f(x):
#         return data.apply(lambda x: np.sign(x) * np.log(abs(x)) if abs(x) > 1 else 0)
#     elif transformation == "log_modified":
#         # Log-modulus transform to preserve 0 values and negative values.
#         # def f(x):
#         return data.apply(lambda x: np.sign(x) * np.log(abs(x) + 1))
#     elif transformation == "yeo-johnson":
#         # def f(x):
#         res, lmbda = yeojohnson(data)
#
#
#     else:
#         def f(x):
#             return x
#
#     return data.apply(f)
#
#
# def post_transformation(data: Series, transformation: str) -> Series:
#     """
#     Applies the inverse of a function (whose name is defined in transformation) to the input data.
#     Returns the transformed data.
#
#     Parameters
#     ----------
#     data : Series
#         Pandas Series. Transformation's inverse will be applied on each value.
#     transformation : str
#         Name of the transformation: the inverse will be applied on the data.
#
#     Returns
#     -------
#     transformed_data : Series
#         Pandas Series where the transformation has been applied.
#     """
#     if transformation == "log":
#         def f(x):
#             return np.sign(x) * np.exp(abs(x))
#     elif transformation == "log_modified":
#         def f(x):
#             return np.sign(x) * np.exp(abs(x)) - np.sign(x)
#     else:
#         def f(x):
#             return x
#
#     return data.apply(f)


def calc_xcorr(target: str, ingested_data: DataFrame, max_lags: int, modes: [str] = ["pearson"]) -> dict:
    """
    Calculate the cross-correlation for the ingested data.
    Use the scenario column as target; the correlation is computed against all lags of all the other columns which
    include numbers. NaN values, introduced by the various shifts, are replaced with 0.

    Parameters
    ----------
    target : str
    Column which is used as target for the cross correlation.

    ingested_data : DataFrame
    Entire dataframe parsed from app

    max_lags : int
    Limit the analysis to max lags.

    modes : [str]
    Cross-correlation can be computed with different algorithms. The available choices are:
        `matlab_normalized`: same as using the MatLab function xcorr(x, y, 'normalized')
        `pearson` : use Pearson formula (NaN values are fillled to 0)
        `kendall`: use Kendall formula (NaN values are filled to 0)
        `spearman`: use Spearman formula (NaN values are filled to 0)

    Returns
    -------
    result : dict
    Dictionary with a Pandas DataFrame set for every indicated mode.
    Each DataFrame has the lags as index and the correlation value for each column.
    """

    def df_shifted(df, _target=None, lag=0):
        if not lag and not _target:
            return df
        new = {}
        for c in df.columns:
            if c == _target:
                new[c] = df[_target]
            else:
                new[c] = df[c].shift(periods=lag)
        return pd.DataFrame(data=new)

    columns = ingested_data.columns.tolist()
    columns = [elem for elem in columns if ingested_data[elem].dtype != str and elem != target]

    results = {}
    for mode in modes:
        result = DataFrame(columns=columns, dtype=np.float64)
        if mode == 'matlab_normalized':
            for col in columns:
                x = ingested_data[target]
                y = ingested_data[col]

                c = np.correlate(x, y, mode="full")

                # This is needed to obtain the same result of the MatLab `xcorr` function with normalized results.
                # You can find the formula in the function pyplot.xcorr; however, here the property
                # sqrt(x*y) = sqrt(x) * sqrt(y)
                # is applied in order to avoid overflows if the ingested values are particularly high.
                den = np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y))
                c = np.divide(c, den)

                # This assigns the correct indexes to the results.
                c = c[len(ingested_data) - 1 - max_lags:len(ingested_data) + max_lags]

                result[col] = c

            result.index -= max_lags

        else:
            for i in range(-max_lags, max_lags + 1):
                shifted = df_shifted(ingested_data, target, i)
                shifted.fillna(0, inplace=True)

                corr = [shifted[target].corr(other=shifted[col], method=mode) for col in columns]
                result.loc[i] = corr

        results[mode] = result

    return results


def calc_all_xcorr(ingested_data: DataFrame, param_config: dict) -> dict:
    """
    Compute, for every column in ingested_data (excluding the index) the cross-correlation of that series with respect
    to all others columns in ingested data.

    Parameters
    ----------
    ingested_data : DataFrame
        Pandas DataFrame for which the cross-correlation of all columns should be computed.

    max_lags : int
        Limit the cross-correlation to at maximum max_lags in the past and future (from -max_lags to max_lags)

    param_config : dict
        TIMEX configuration dictionary, needed to for xcorr_modes and xcorr_max_lags.
        xcorr_modes indicate the different algorithms which should be used to compute the xcorr.
        The available choices are:
            `matlab_normalized`: same as using the MatLab function xcorr(x, y, 'normalized')
            `pearson` : use Pearson formula (NaN values are fillled to 0)
            `kendall`: use Kendall formula (NaN values are filled to 0)
            `spearman`: use Spearman formula (NaN values are filled to 0)

        xcorr_max_lags is the number of lags, both in positive and negative direction, to which the cross-correlation
        calculations should be limited to.

    Returns
    -------
    dict
        Python dict with a key for every cross-correlation algorithm requested.
    """
    xcorr_max_lags = param_config['model_parameters']['xcorr_max_lags']
    xcorr_modes = [*param_config['model_parameters']["xcorr_mode"].split(",")]
    d = {}
    for col in ingested_data.columns:
        d[col] = calc_xcorr(col, ingested_data, max_lags=xcorr_max_lags, modes=xcorr_modes)

    return d
