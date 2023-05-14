import logging

from pandas import DataFrame
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

from timexseries.data_prediction import PredictionModel
from timexseries.data_prediction.models.seasonality_estimator import estimate_seasonality

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

        seasonality = estimate_seasonality(train_data)

        seasonality = min(24, seasonality)

        train_data.reset_index(inplace=True)
        train_data.columns = ['ds', 'y']
        train_data.loc[:, 'unique_id'] = 0

        # if seasonality == 1:
        #     model = 'ZZN'
        # else:
        #     model = 'ZZA'

        model = StatsForecast(
            df=train_data,
            models=[AutoETS(season_length=seasonality, model='ZZZ')],
            freq=freq
        )

        y_hat_df = model.forecast(points_to_predict, level=[95]).set_index("ds")

        future_dataframe.iloc[-points_to_predict:, 0] = y_hat_df.loc[:, 'AutoETS']
        index_to_update = future_dataframe.index[-points_to_predict:]
        future_dataframe.loc[index_to_update, 'yhat_upper'] = y_hat_df.loc[:, 'AutoETS-hi-95']
        future_dataframe.loc[index_to_update, 'yhat_lower'] = y_hat_df.loc[:, 'AutoETS-lo-95']

        return future_dataframe
