import abc
from math import sqrt

from pandas import DataFrame
from sklearn.metrics import mean_squared_error, mean_absolute_error


class PredictionModel(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'train') and
                callable(subclass.train) and
                hasattr(subclass, 'predict') and
                callable(subclass.predict) and
                hasattr(subclass, 'get_training_parameters') and
                callable(subclass.get_training_parameters))


def get_training_stats(actual: DataFrame, predicted: DataFrame) -> dict:
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    return {
        'MSE': mse,
        'RMSE': sqrt(mse),
        'MAE': mae
    }


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
