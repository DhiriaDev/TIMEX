from statsforecast import StatsForecast
from timexseries.data_prediction import PredictionModel
from statsforecast.models import SeasonalNaive
import pandas as pd


class SeasonalPersistenceModel(PredictionModel):
    """Seasonal persistence (naive) prediction model."""
    def predict(self, train_ts: pd.Series, seasonality: int, forecast_horizon: int) -> pd.Series:

        freq = train_ts.index.freq
        name = train_ts.name

        train_ts = pd.DataFrame(train_ts)
        train_ts.reset_index(inplace=True)
        train_ts.columns = ['ds', 'y']
        train_ts.loc[:, 'unique_id'] = 0

        model = StatsForecast(
            df=train_ts,
            models=[SeasonalNaive(season_length=seasonality)],
            freq=freq
        )

        y_hat_df = model.forecast(forecast_horizon).set_index("ds")
        y_hat = y_hat_df.loc[:, 'SeasonalNaive']
        y_hat.name = name
        return y_hat

