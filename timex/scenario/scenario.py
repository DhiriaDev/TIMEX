from pandas import DataFrame

from timex.data_prediction.data_prediction import ModelResult


class Scenario:
    """
    A scenario represents a full analysis of a single univariate time series.
    It includes all the relevant information needed to characterize the study case.

    Parameters
    ----------
    scenario_data : DataFrame
        Initial time series, in the form of a DataFrame with a index and a single data column.
    models : [ModelResult]
        List of ModelResult objects, all trained on this scenario.
    ingested_data : DataFrame
        The entire DataFrame parsed starting from the CSV. Useful for cross-correlation diagrams.
    xcorr : dict
        Cross-correlation between the data of this scenario and all the other ones.
    """
    def __init__(self, scenario_data: DataFrame, models: [ModelResult], ingested_data: DataFrame, xcorr: dict):
        self.scenario_data = scenario_data
        self.models = models
        self.ingested_data = ingested_data
        self.xcorr = xcorr
