from pandas import DataFrame

from statsforecast import StatsForecast
from timexseries.data_prediction import PredictionModel
from statsforecast.models import AutoARIMA, SeasonalNaive, RandomWalkWithDrift

from timexseries.data_prediction.models.seasonality_estimator import estimate_seasonality


class RandomWalkWithDriftModel(PredictionModel):
    """Random walk with drift prediction model."""
    def __init__(self, params: dict, transformation: str = "none"):
        super().__init__(params, name="Random walk with drift", transformation=transformation)

    def predict(self, train_data: DataFrame, points_to_predict: int,
                future_dataframe: DataFrame, extra_regressor: DataFrame = None) -> DataFrame:
        freq = train_data.index.freq

        train_data.reset_index(inplace=True)
        train_data.columns = ['ds', 'y']
        train_data.loc[:, 'unique_id'] = 0

        model = StatsForecast(
            df=train_data,
            models=[RandomWalkWithDrift()],
            freq=freq
        )

        y_hat_df = model.forecast(points_to_predict).set_index("ds")
        future_dataframe.iloc[-points_to_predict:, 0] = y_hat_df.loc[:, 'RWD']

        return future_dataframe
