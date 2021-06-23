from pandas import DataFrame
import numpy as np

from timexseries.data_prediction import PredictionModel
import pmdarima as pm


class ARIMAModel(PredictionModel):
    """ARIMA prediction model."""
    def __init__(self, params: dict, transformation: str = "none"):
        super().__init__(params, name="ARIMA", transformation=transformation)
        self.arima = None
        self.len_train_set = 0

    def train(self, input_data: DataFrame, extra_regressor: DataFrame = None):
        # This is the code needed to check if a SARIMA is needed.
        # However, fitting a SARIMA can take a huge amount of time.
        # s = pd.Series(sm.tsa.acf(input_data, nlags=round(len(input_data) / 2 - 1), fft=True))
        #
        # z99 = 2.5758293035489004
        # len_series = len(input_data)
        # threshold99 = z99 / np.sqrt(len_series)
        #
        # peaks, _ = find_peaks(s, height=threshold99, prominence=0.01)
        # s = s[peaks].sort_values(ascending=False)
        #
        # if len(s) > 0:
        #     # Possible seasonality.
        #     self.arima = pm.auto_arima(input_data, error_action='ignore', seasonal=True, suppress_warnings=True,
        #                                m=s.index[0])
        # else:
        self.arima = pm.auto_arima(input_data, error_action='ignore', seasonal=False, suppress_warnings=True)

        self.len_train_set = len(input_data)

    def predict(self, future_dataframe: DataFrame, extra_regressor: DataFrame = None) -> DataFrame:
        """Overrides PredictionModel.predict()"""
        requested_prediction = len(future_dataframe) - self.len_train_set
        predictions = self.arima.predict_in_sample()
        predictions = np.concatenate((predictions, self.arima.predict(n_periods=requested_prediction)))

        future_dataframe.yhat = predictions

        return future_dataframe
