import logging
import warnings

from pandas import DataFrame
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsforecast import StatsForecast
from statsforecast.models import ETS
from statsmodels.tsa.stattools import adfuller

from timexseries.data_prediction import PredictionModel
import numpy as np
import pandas as pd
import statsmodels.api as sm

log = logging.getLogger(__name__)


class ExponentialSmoothingModel(PredictionModel):
    """
    Exponential smoothing prediction model.
    This model uses the Holt Winters method and automatic parameters estimation.
    """

    def __init__(self, params: dict, transformation: str = "none"):
        super().__init__(params, name="ExponentialSmoothing", transformation=transformation)
        self.model = None
        self.len_train_set = 0
        self._forecast_horizon = 0

    def train(self, input_data: DataFrame, points_to_predict: int, extra_regressors: DataFrame = None):
        """Overrides PredictionModel.train()"""
        self._forecast_horizon = points_to_predict
        freq = input_data.index.freq

        s = pd.Series(sm.tsa.acf(input_data, nlags=round(len(input_data) / 2 - 1), fft=True))
        s[0] = 0
        s[1] = 0

        z99 = 2.5758293035489004
        lenSeries = len(input_data)
        threshold99 = z99 / np.sqrt(lenSeries)

        s = s[lambda x: x > threshold99].sort_values(ascending=False)
        if len(s) > 0:
            seasonality = s.index[0]
            model = "ZAZ"
        else:
            seasonality = 1
            model = "ZNZ"

        input_data.reset_index(inplace=True)
        input_data.columns = ['ds', 'y']
        input_data.loc[:, 'unique_id'] = 0

        try:
            models = [
                ETS(season_length=seasonality, model=model)
            ]

            self.model = StatsForecast(
                df=input_data,
                models=models,
                freq=freq
            )

            self.Y_hat_df = self.model.forecast(self._forecast_horizon).set_index("ds")
        except:
            models = [
                ETS(season_length=1, model='ZNZ')
            ]

            self.model = StatsForecast(
                df=input_data,
                models=models,
                freq=freq
            )

            self.Y_hat_df = self.model.forecast(self._forecast_horizon).set_index("ds")

    def predict(self, future_dataframe: DataFrame, extra_regressors: DataFrame = None) -> DataFrame:
        """Overrides PredictionModel.predict()"""
        future_dataframe.loc[-self._forecast_horizon:, 'yhat'] = self.Y_hat_df.loc[:, 'ETS']

        return future_dataframe
