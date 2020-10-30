from pandas import DataFrame

from src.data_prediction.data_prediction import ModelResult


class Scenario:
    """
    A scenario represents a full analysis of a single univariate time series.
    It includes all the relevant information needed to characterize the study case.

    Parameters
    ----------
    ingested_data : DataFrame
        Initial time series, in the form of a DataFrame with a index and a single data column.
    model_results : [ModelResult]
        List of ModelResult objects, all trained on this scenario.
    """
    def __init__(self, ingested_data: DataFrame, model_results: [ModelResult]):
        self.ingested_data = ingested_data
        self.model_results = model_results
