import logging

from pandas import DataFrame

from timexseries.data_prediction import PredictionModel
log = logging.getLogger(__name__)


class PersistenceModel(PredictionModel):
    """
    Persistence model predictor (also called *naive* model).
    Its prediction is to use the last known value.
    """

    def __init__(self, params: dict, transformation: str = "none"):
        super().__init__(params, name="Persistence", transformation=transformation)

        self.last_known_value = 0
        self.len_train_set = 0
        self.requested_predictions = 0

    def train(self, input_data: DataFrame, points_to_predict: int, extra_regressors: DataFrame = None):
        """Overrides PredictionModel.train()"""
        self.last_known_value = input_data.iloc[-1, 0]
        self.points_to_predict = points_to_predict

    def predict(self, future_dataframe: DataFrame, extra_regressors: DataFrame = None) -> DataFrame:
        """Overrides PredictionModel.predict()"""
        future_dataframe.iloc[-self.points_to_predict:, 0] = self.last_known_value
        return future_dataframe


