from math import sqrt

from pandas import DataFrame
from sklearn.metrics import mean_squared_error, mean_absolute_error


class ValidationPerformance:
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