from pandas import DataFrame


class Scenario:
    """
    A scenario represents a full analysis of a single univariate time series.
    It includes all the relevant information needed to characterize the study case.

    Parameters
    ----------
    scenario_data : DataFrame
        Initial time series, in the form of a DataFrame with a index and a single data column.
    models : dict
        Dict of ModelResult objects, all trained on this scenario.
    xcorr : dict
        Cross-correlation between the data of this scenario and all the other ones.
    """
    def __init__(self, scenario_data: DataFrame, models: dict, xcorr: dict, historical_prediction: dict = None):
        self.scenario_data = scenario_data
        self.models = models
        self.xcorr = xcorr
        self.historical_prediction = historical_prediction

    def set_historical_prediction(self, historical_prediction):
        self.historical_prediction = historical_prediction
