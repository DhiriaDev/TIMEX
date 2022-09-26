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

    def predict(self, train_data: DataFrame, points_to_predict: int,
                future_dataframe: DataFrame, extra_regressors: DataFrame = None) -> DataFrame:
        """Overrides PredictionModel.predict()"""
        future_dataframe.iloc[-points_to_predict:, 0] = train_data.iloc[-1, 0]
        return future_dataframe
