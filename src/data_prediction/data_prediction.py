import abc
import json
import math
import pkgutil
from math import sqrt

import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


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
    """

    def __init__(self, first_used_index):
        self.first_used_index = first_used_index
        self.MSE = 0
        self.RMSE = 0
        self.MAE = 0

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


class ModelResult:
    """
    Class for the result of a model trained on a time series.
    This will include predicted data, performance indexes and model
    parameters, able to fully characterize the model itself.

    Attributes
    ----------
    prediction : DataFrame
        DataFrame contained the values predicted by the trained model.
    testing_performances : [TestingPerformance]
        Performance achieved by the model.
    characteristics : dict
        Model parameters, obtained by automatic tuning.
    """

    def __init__(self, prediction: DataFrame, testing_performances: [TestingPerformance], characteristics: dict):
        self.prediction = prediction
        self.testing_performances = testing_performances
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

    def __init__(self, params: dict, name: str) -> None:
        self.name = name
        self.verbose = params["verbose"]

        if params["verbose"] == "yes":
            print('-----------------------------------------------------------')
            print('Model_training: creating ' + self.name + " model...")

        if "model_parameters" not in params:
            if params["verbose"] == "yes":
                print("Model_training: loading default settings...")
            parsed = pkgutil.get_data(__name__, "default_prediction_parameters/" + self.name + ".json")
            model_parameters = json.loads(parsed)
        else:
            if params["verbose"] == "yes":
                print("Model_training: loading user settings...")
            model_parameters = params["model_parameters"]

        self.test_percentage = model_parameters["test_percentage"]
        self.prediction_lags = model_parameters["prediction_lags"]
        self.transformation = model_parameters["transformation"]
        self.delta_training_percentage = model_parameters["delta_training_percentage"]
        self.main_accuracy_estimator = model_parameters["main_accuracy_estimator"]
        self.delta_training_values = 0
        self.model_characteristics = {}

        self.test_values = 0
        self.freq = ""

    def train(self, ingested_data: DataFrame) -> TestingPerformance:
        """
        Train the model on the ingested_data.
        Returns the statistical performance of the testing.

        Parameters
        ----------
        ingested_data : DataFrame
            DataFrame which _HAS_ to be divided in training and test set.

        Returns
        -------
        testing_performance : TestingPerformance
        """
        pass

    def predict(self) -> DataFrame:
        """
        Return a DataFrame with the length of the one used for training
        plus self.prediction_lags, with predicted values.

        Returns
        -------
        forecast : DataFrame
        """
        pass

    def launch_model(self, ingested_data: DataFrame) -> ModelResult:
        """
        Train the model on ingested_data and returns a ModelResult object.

        Returns
        -------
        model_result : ModelResult
            Object containing the results of the model, trained on ingested_data.
        """
        model_characteristics = self.model_characteristics

        self.delta_training_values = int(round(len(ingested_data) * self.delta_training_percentage / 100))

        test_values = int(round(len(ingested_data) * (self.test_percentage / 100)))
        self.test_values = test_values
        self.freq = pd.infer_freq(ingested_data.index)

        train_ts = ingested_data.iloc[:-test_values]
        test_ts = ingested_data.iloc[-test_values:]

        train_sets_number = math.floor(len(train_ts) / self.delta_training_values) + 1

        with pd.option_context('mode.chained_assignment', None):
            train_ts.iloc[:, 0] = pre_transformation(train_ts.iloc[:, 0], self.transformation)

        results = []

        if self.verbose == "yes":
            print("Model will use " + str(train_sets_number) + " different training sets.")
        for i in range(1, train_sets_number):
            tr = train_ts.iloc[-i * self.delta_training_values:]

            if self.verbose == "yes":
                print("Trying with last " + str(len(tr)) + " values as training set...")

            self.train(tr.copy())

            forecast = self.predict()
            testing_prediction = forecast.iloc[-self.prediction_lags - test_values:-self.prediction_lags]

            first_used_index = tr.index.values[0]

            tp = TestingPerformance(first_used_index)
            tp.set_testing_stats(test_ts.iloc[:, 0], testing_prediction["yhat"])
            results.append({
                "testing_performances": tp,
                "forecast": forecast,
            })

        results.sort(key=lambda x: getattr(x["testing_performances"], self.main_accuracy_estimator.upper()))
        best_forecast = results[0]["forecast"]
        testing_results = [x["testing_performances"] for x in results]

        model_characteristics["name"] = self.name
        model_characteristics["delta_training_percentage"] = str(self.delta_training_percentage) + "%"
        model_characteristics["delta_training_values"] = str(self.delta_training_values)

        return ModelResult(prediction=best_forecast,
                           testing_performances=testing_results,
                           characteristics=model_characteristics)


def pre_transformation(data: Series, transformation: str) -> Series:
    """
    Applies a function (whose name is defined in transformation) to the input data.
    Returns the transformed data.

    Parameters
    ----------
    data : Series
        Pandas Series. Transformation will be applied to each value.
    transformation : str
        Name of the transformation which should be applied.

    Returns
    -------
    transformed_data : Series
        Series where the transformation has been applied.
    """
    if transformation == "log":
        def f(x):
            return np.log(x) if x > 0 else 0
        return data.apply(f)
    else:
        return data


def post_transformation(data: Series, transformation: str) -> Series:
    """
    Applies the inverse of a function (whose name is defined in transformation) to the input data.
    Returns the transformed data.

    Parameters
    ----------
    data : Series
        Pandas Series. Transformation's inverse will be applied on each value.
    transformation : str
        Name of the transformation: the inverse will be applied on the data.

    Returns
    -------
    transformed_data : Series
        Pandas Series where the transformation has been applied.
    """
    if transformation == "log":
        f = np.exp
    else:
        f = lambda x: x

    return f(data)


