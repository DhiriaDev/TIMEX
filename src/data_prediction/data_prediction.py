import abc
from math import sqrt

from pandas import DataFrame
from sklearn.metrics import mean_squared_error, mean_absolute_error


class TrainingPerformance:
    """
    Class for the summary of various statistical indexes relative
    to the performance of a prediction model.

    Attributes
    ----------
    MSE : float
        Mean Squared Error. Default 0
    RMSE: float
        Root Mean Squared Error. Default 0
    MAE: float
        Mean Absolute Error. Default 0
    """

    def __init__(self):
        self.MSE = 0
        self.RMSE = 0
        self.MAE = 0

    def set_training_stats(self, actual: DataFrame, predicted: DataFrame):
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
    training_performance : TrainingPerformance
        Performance achieved by the model.
    characteristics : dict
        Model parameters, obtained by automatic tuning.
    """
    def __init__(self, prediction: DataFrame, training_performance: TrainingPerformance, characteristics: dict):
        self.prediction = prediction
        self.training_performance = training_performance
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
    pre_transformation : callable
        Transformation to apply to the time series before using it. Default None
    post_transformation : callable
        Inverse of pre_transformation. Default None
    prediction_lags : int
        Number of future lags for which the prediction has to be made. Default 0
    model_characteristics : dict
        Dictionary of values containing the main characteristics and parameters
        of the model. Default {}
    """

    def __init__(self) -> None:
        self.freq = None
        self.test_percentage = 0
        self.test_values = 0
        self.pre_transformation = None
        self.post_transformation = None
        self.prediction_lags = 0
        self.model_characteristics = {}

    def train(self, input_data: DataFrame) -> TrainingPerformance:
        """
        Train the model on the input_data.
        Returns the statistical performance of the training.

        Parameters
        ----------
        input_data : DataFrame
            DataFrame which _HAS_ to be divided in training and test set.

        Returns
        -------
        training_performance : TrainingPerformance
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
        Train the model on ingested_data and returns a model_result dict.

        Returns
        -------
        model_result : ModelResult
            Object containing the results of the model, trained on ingested_data.
        """
        training_performance = self.train(ingested_data)
        prediction = self.predict()
        model_characteristics = self.model_characteristics

        return ModelResult(prediction, training_performance, model_characteristics)


class ARIMA:
    """ARIMA prediction model."""

    def __init__(self):
        pass

    def train(self):
        """Overrides PredictionModel.train()"""
        pass

    def predict(self):
        """Overrides PredictionModel.predict()"""
        pass

    def get_training_parameters(self):
        """Overrides PredictionModel.get_training_parameters()"""
        pass
