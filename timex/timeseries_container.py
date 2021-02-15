from pandas import DataFrame


class TimeSeriesContainer:
    """
    A TimeSeriesContainer represents a full analysis of a single univariate time series.
    It includes all the relevant information needed to characterize the study case.

    Parameters
    ----------
    timeseries_data : DataFrame
        Initial time series, in the form of a DataFrame with a index and a single data column.
    models : dict
        Dict of ModelResult objects, all trained on this time-series.
    xcorr : dict
        Cross-correlation between the data of this time-series and all the other ones.
    """
    def __init__(self, timeseries_data: DataFrame, models: dict, xcorr: dict, historical_prediction: dict = None):
        self.timeseries_data = timeseries_data
        self.models = models
        self.xcorr = xcorr
        self.historical_prediction = historical_prediction

    def set_historical_prediction(self, historical_prediction):
        self.historical_prediction = historical_prediction
