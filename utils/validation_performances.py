from math import sqrt

from pandas import Series
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ValidationPerformance:
    """
    Class for the summary of various statistical indexes relative to the performance of a prediction model.

    Parameters
    ----------
    first_used_index, optional, default None
        Index of the first value used in the time series to generate these results. This is a convenience if you already
        know that initial index of the data this performance will refer to.

    Attributes
    ----------
    R2 : float
        The stationary R-squared to compare the stationary part of the model to a simple mean model. Default 0
    RMSE: float
        Root Mean Squared Error. Default 0
    MAE: float
        Mean Absolute Error. Default 0
    AM: float
        Arithmetic Mean of error. Default 0
    SD: float
        Standard deviation of error. Default 0
    """

    def __init__(self, first_used_index=None):
        self.first_used_index = first_used_index
        self.R2 = 0
        self.RMSE = 0
        self.MAE = 0
        self.AM = 0
        self.SD = 0



    def set_testing_stats(self, actual: Series, predicted: Series):
        """
        Set all the statistical indexes according to input data.

        Parameters
        ----------
        actual : Series
            Actual data stored in a Pandas Series.
        predicted : Series
            Data predicted by a model, stored in a Pandas Series.

        Examples
        --------
        >>> dates = pd.date_range('2000-01-01', periods=5)
        >>> ds = pd.DatetimeIndex(dates, freq="D")
        >>> actual = np.array([1, 1, 1, 1, 1])
        >>> predicted = np.array([3, 3, 3, 3, 3])
        >>> actual_dataframe = DataFrame(data={"a": actual}, index=ds)
        >>> predicted_dataframe = DataFrame(data={"yhat": predicted}, index=ds)

        Calculate the performances.
        >>> perf = ValidationPerformance()
        >>> perf.set_testing_stats(actual_dataframe['a'], predicted_dataframe['yhat'])

        >>> print(perf.MAE)
        2.0

        >>> print(perf.RMSE)
        2.0
        """
        self.R2 = r2_score(actual, predicted)
        self.MAE = mean_absolute_error(actual, predicted)
        self.RMSE = sqrt(mean_squared_error(actual, predicted))
        self.AM = sum([y - yhat for y, yhat in zip(actual, predicted)]) / len(actual)
        self.SD = (actual - predicted).std(ddof=0)
    
    @staticmethod
    def get_available_metrics() -> list[str]:
        return ['R2', 'RMSE', 'MAE', 'AM', 'SD']
    
    def get_dict(self) -> dict:
        """
        Return all the parameters, in a dict.

        Returns
        -------
        d : dict
            All the statistics, in a dict.

        Examples
        --------
        >>> perf = ValidationPerformance()
        >>> perf.set_testing_stats(actual_dataframe['a'], predicted_dataframe['yhat'])
        >>> perf.get_dict()
        {'first_used_index': None, 'RMSE': 4.0, 'RMSE': 2.0, 'MAE': 2.0, 'AM': -2.0, 'SD': 0.0}
        """
        d = {}
        for attribute, value in self.__dict__.items():
            d[attribute] = value

        return d