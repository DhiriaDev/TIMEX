from pandas import DataFrame


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
        tested.
    """
    def __init__(self, timeseries_data: DataFrame, models: dict, xcorr: dict, historical_prediction: dict = None):
        self.timeseries_data = timeseries_data
        self.models = models
        self.xcorr = xcorr
        self.historical_prediction = historical_prediction

    def set_historical_prediction(self, historical_prediction):
        self.historical_prediction = historical_prediction
