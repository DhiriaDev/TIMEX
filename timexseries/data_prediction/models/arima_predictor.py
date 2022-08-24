from pandas import DataFrame
import numpy as np

from timexseries.data_prediction import PredictionModel
import pmdarima as pm


class ARIMAModel(PredictionModel):
    """ARIMA prediction model."""
    def __init__(self, params: dict, transformation: str = "none"):
        super().__init__(params, name="ARIMA", transformation=transformation)

    def predict(self, train_data: DataFrame, points_to_predict: int,
                future_dataframe: DataFrame, extra_regressor: DataFrame = None) -> DataFrame:
        """Overrides PredictionModel.predict()"""
        arima = pm.auto_arima(train_data, error_action='ignore', seasonal=False, suppress_warnings=True)
        predictions = arima.predict_in_sample()
        predictions = np.concatenate((predictions, arima.predict(n_periods=points_to_predict)))

        future_dataframe.yhat = predictions

        return future_dataframe
