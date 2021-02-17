import logging

from pandas import DataFrame

from timexseries.data_prediction import PredictionModel
log = logging.getLogger(__name__)


class MockUpModel(PredictionModel):
    """
    Mock up prediction model. Useful for testing purposes.
    This model "simulates" a real model by returning a - at least in dimensions - correct dataframe.
    This dataframe predicts always 0 is no extra regressors have been given, 1 otherwise.

    This can be useful in tests because it runs in very low time and is useful to understand if higher-level functions
    work as intended.
    """

    def __init__(self, params: dict, transformation: str = None):
        super().__init__(params, name="MockUp", transformation=transformation)
        self.extra_regressors_in_training = None
        self.extra_regressors_in_predict = None
        self.len_train_set = 0
        self.requested_predictions = 0

    def train(self, input_data: DataFrame, extra_regressors: DataFrame = None):
        """Overrides PredictionModel.train()"""
        self.extra_regressors_in_training = extra_regressors
        self.len_train_set = len(input_data)

    def predict(self, future_dataframe: DataFrame, extra_regressors: DataFrame = None) -> DataFrame:
        """Overrides PredictionModel.predict()"""
        self.extra_regressors_in_training = extra_regressors
        self.requested_predictions = len(future_dataframe) - self.len_train_set
        self.extra_regressors_in_predict = extra_regressors

        if extra_regressors is None:
            future_dataframe.iloc[-self.requested_predictions:, 0] = 0.0
        else:
            future_dataframe.iloc[-self.requested_predictions:, 0] = len(extra_regressors.columns)

        return future_dataframe.copy()


