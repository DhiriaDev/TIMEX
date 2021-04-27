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

    def __init__(self, params: dict, transformation: str = "none"):
        super().__init__(params, name="MockUp", transformation=transformation)
        try:
            if params["model_parameters"]["mockup_confidence"]:
                self.confidence_intervals = True
            else:
                self.confidence_intervals = False
        except KeyError:
            self.confidence_intervals = False

        try:
            self.forced_predictions = params["model_parameters"]["mockup_forced_predictions"]
        except KeyError:
            self.forced_predictions = None

        # try:
        #     self.forecast_value = params["model_parameters"]["mockup_forecast_value"]
        # except KeyError:
        #     self.forecast_value = 0

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

        if self.forced_predictions is not None:
            initial_index = future_dataframe.index[0]
            final_index = future_dataframe.index[-1]
            future_dataframe.loc[:, 'yhat'] = self.forced_predictions.loc[initial_index:final_index]
            if self.confidence_intervals:
                future_dataframe.loc[:, 'yhat_lower'] = self.forced_predictions.loc[initial_index:final_index].apply(
                    lambda x: x - 0.5
                )
                future_dataframe.loc[:, 'yhat_upper'] = self.forced_predictions.loc[initial_index:final_index].apply(
                    lambda x: x + 0.5
                )
        else:
            if extra_regressors is None:
                v = 0
            else:
                v = len(extra_regressors.columns)

            future_dataframe.loc[future_dataframe.index[-self.requested_predictions]:, 'yhat'] = v
            if self.confidence_intervals:
                future_dataframe.loc[future_dataframe.index[-self.requested_predictions]:, 'yhat_lower'] = v - 0.5
                future_dataframe.loc[future_dataframe.index[-self.requested_predictions]:, 'yhat_upper'] = v + 0.5

        return future_dataframe.copy()


