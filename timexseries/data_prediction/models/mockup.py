from timexseries.data_prediction import PredictionModel
import pandas as pd


class MockUp(PredictionModel):
    def predict(self, train_ts: pd.Series, seasonality: int, forecast_horizon: int) -> pd.Series:

        freq = train_ts.index.freq
        last_index = train_ts.index[-1]
        name = train_ts.name

        index = pd.date_range(last_index + 1 * freq,
                              end=last_index + forecast_horizon * freq)
        y_hat = pd.Series([0 for _ in range(0, forecast_horizon)], index=index)

        y_hat.name = name
        return y_hat

