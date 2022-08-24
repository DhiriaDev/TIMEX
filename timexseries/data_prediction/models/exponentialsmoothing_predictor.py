import logging

from pandas import DataFrame
from statsforecast import StatsForecast
from statsforecast.models import ETS

from timexseries.data_prediction import PredictionModel
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

    def predict(self, train_data: DataFrame, points_to_predict: int, 
                future_dataframe: DataFrame, extra_regressors: DataFrame = None) -> DataFrame:
        """Overrides PredictionModel.predict()"""
        freq = train_data.index.freq

        try:
            s, confidence_intervals = sm.tsa.pacf(train_data.diff(1)[1:], method='ywm', nlags=int(len(train_data) / 2)-5,
                                                  alpha=0.01)
            s, confidence_intervals = pd.Series(abs(s)), pd.Series([abs(x[0]) for x in confidence_intervals])
            s[0] = 0
            s[1] = 0

            s = s[s > confidence_intervals].sort_values(ascending=False)
            if len(s) > 0:
                seasonality = s.index[0]
            else:
                seasonality = 0
        except:  # LinAlgError
            seasonality = 0

        train_data.reset_index(inplace=True)
        train_data.columns = ['ds', 'y']
        train_data.loc[:, 'unique_id'] = 0

        if seasonality == 0:
            models = [
                ETS(season_length=1, model='ZNZ')
            ]

            model = StatsForecast(
                df=train_data,
                models=models,
                freq=freq
            )

            y_hat_df = model.forecast(points_to_predict).set_index("ds")
        else:
            try:
                models = [
                    ETS(season_length=seasonality, model='ZZZ')
                ]

                model = StatsForecast(
                    df=train_data,
                    models=models,
                    freq=freq
                )

                y_hat_df = model.forecast(points_to_predict).set_index("ds")
            except:
                models = [
                    ETS(season_length=1, model='ZNZ')
                ]

                model = StatsForecast(
                    df=train_data,
                    models=models,
                    freq=freq
                )

                y_hat_df = model.forecast(points_to_predict).set_index("ds")

        future_dataframe.iloc[-points_to_predict:, 0] = y_hat_df.loc[:, 'ETS']

        return future_dataframe
