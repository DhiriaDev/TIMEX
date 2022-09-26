from pandas import DataFrame

from statsforecast import StatsForecast
from .. import PredictionModel
from statsforecast.models import AutoARIMA

from .seasonality_estimator import estimate_seasonality


class ARIMAModel(PredictionModel):
    """ARIMA prediction model."""
    def __init__(self, params: dict, transformation: str = "none"):
        super().__init__(params, name="ARIMA", transformation=transformation)

    def predict(self, train_data: DataFrame, points_to_predict: int,
                future_dataframe: DataFrame, extra_regressor: DataFrame = None) -> DataFrame:
        freq = train_data.index.freq

        seasonality = estimate_seasonality(train_data)

        train_data.reset_index(inplace=True)
        train_data.columns = ['ds', 'y']
        train_data.loc[:, 'unique_id'] = 0

        model = StatsForecast(
            df=train_data,
            models=[AutoARIMA(season_length=seasonality, seasonal=seasonality != 1)],
            freq=freq
        )

        y_hat_df = model.forecast(points_to_predict).set_index("ds")

        future_dataframe.iloc[-points_to_predict:, 0] = y_hat_df.loc[:, 'AutoARIMA']

        return future_dataframe
