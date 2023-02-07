import pandas as pd

from timexseries.data_prediction import PredictionModel
from statsforecast import StatsForecast
from statsforecast.models import ETS as statsforecastETS


class ETS(PredictionModel):
    """
    Exponential smoothing prediction model.
    This model uses the Holt Winters method and automatic parameters estimation.
    """

    def __init__(self):
        super().__init__()

    def predict(self, train_ts: pd.Series, seasonality: int, points_to_predict: int) -> pd.Series:
        """Overrides PredictionModel.predict()"""
        freq = train_ts.index.freq
        name = train_ts.name

        seasonality = min(24, seasonality)

        train_ts = pd.DataFrame(train_ts)
        train_ts.reset_index(inplace=True)
        train_ts.columns = ['ds', 'y']
        train_ts.loc[:, 'unique_id'] = 0

        if seasonality == 1:
            model = 'ZZN'
        else:
            model = 'ZZA'

        model = StatsForecast(
            df=train_ts,
            models=[statsforecastETS(season_length=seasonality, model=model)],
            freq=freq
        )

        try:
            y_hat_df = model.forecast(points_to_predict).set_index("ds")
            y_hat = y_hat_df.loc[:, 'ETS']
            y_hat.name = name
            return y_hat
        except:
            last_value = train_ts.loc[:, 'y'].iloc[-1]
            last_index = train_ts.loc[:, 'ds'].iloc[-1] + 1 * freq
            y_hat_df = pd.Series([last_value for _ in range(0, points_to_predict)],
                                 index=pd.date_range(last_index, periods=points_to_predict, freq=freq, name=name))
            return y_hat_df