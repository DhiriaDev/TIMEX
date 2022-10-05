"""
.. include:: ./documentation.md
"""
from math import sqrt

import numpy as np
import pandas as pd

from pandas import DataFrame, Series


class TimeSeriesContainer:
    """
    A TimeSeriesContainer collect all the relevant information useful to characterize a single time-series coming from
    the ingested dataset.

    Parameters
    ----------
    timeseries_data : DataFrame
        Historical time-series data, in the form of a DataFrame with a index and a single data column.
    models : dict
        Dictionary of ModelResult objects, all trained on this time-series.
    xcorr : dict
        Cross-correlation between the data of this time-series and all the other ones.
    historical_prediction : dict
        The historical prediction, i.e. the predictions computed on a rolling window on the historical data.
        This is useful to verify the performances of each model not only on the very last data, but throughout the
        history of the time-series, in a cross-validation fashion. This dictionary contains one entry for each model
        tested, and each of this entry contains two keys: 'series', i.e., the actual historical prediction, and
        'metrics', i.e. validation performances of the historical predictions on real data.
    """
    def __init__(self, timeseries_data: DataFrame, models: dict, xcorr: dict, historical_prediction: dict = None):
        self.timeseries_data = timeseries_data
        self.models = models
        self.xcorr = xcorr
        self.historical_prediction = historical_prediction

    def set_historical_prediction(self, historical_prediction):
        self.historical_prediction = historical_prediction


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
    MSE : float
        Mean Squared Error. Default 0
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
        self.MSE = 0
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

        >>> print(perf.MSE)
        4.0
        """
        self.MSE = np.square(np.subtract(actual,predicted)).mean()
        self.MAE = np.abs(np.subtract(actual, predicted)).mean()

        self.RMSE = sqrt(self.MSE)
        self.AM = sum([y - yhat for y, yhat in zip(actual, predicted)]) / len(actual)
        self.SD = (actual - predicted).std(ddof=0)

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
        {'first_used_index': None, 'MSE': 4.0, 'RMSE': 2.0, 'MAE': 2.0, 'AM': -2.0, 'SD': 0.0}
        """
        d = {}
        for attribute, value in self.__dict__.items():
            d[attribute] = value

        return d


class SingleResult:
    """
    Class for the result of a model, trained on a specific training set.

    Parameters
    ----------
    prediction : DataFrame
        Estimated prediction, using this training set
    testing_performances : ValidationPerformance
        Testing performance (`timexseries.data_prediction.validation_performances.ValidationPerformance`), on the validation
        set, obtained using this training set to train the model.
    """

    def __init__(self, prediction: DataFrame, testing_performances: ValidationPerformance):
        self.prediction = prediction
        self.testing_performances = testing_performances


class ModelResult:
    """
    Class for to collect the global results of a model trained on a time-series.

    Parameters
    ----------
    results : [SingleResult]
        List of all the results obtained using all the possible training set for this model, on the time series.
        This is useful to create plots which show how the performance vary changing the training data (e.g.
        `timexseries.data_visualization.functions.performance_plot`).
    characteristics : dict
        Model parameters. This dictionary collects human-readable characteristics of the model, e.g. the used number of
        validation points used, the length of the sliding training window, etc.
    best_prediction : DataFrame
        Prediction obtained using the best training window and _all_ the available points in the time-series. This is
        the prediction that users are most likely to want.
    """

    def __init__(self, results: [SingleResult], characteristics: dict, best_prediction: DataFrame):
        self.results = results
        self.characteristics = characteristics
        self.best_prediction = best_prediction


