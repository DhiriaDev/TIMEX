import warnings

import numpy as np
from pandas import DataFrame
from timexseries.data_prediction import PredictionModel
from sklearn.linear_model import LinearRegression


class LinearModel(PredictionModel):
    """Linear prediction model."""
    def __init__(self, params: dict, transformation: str = "Linear"):
        super().__init__(params, name="Linear", transformation=transformation)

    def predict(self, train_data: DataFrame, points_to_predict: int,
                future_dataframe: DataFrame, extra_regressor: DataFrame = None) -> DataFrame:

        train_data.reset_index(inplace=True)
        train_data.columns = ['ds', 'y']

        model = LinearRegression()
        model.fit(np.arange(1, len(train_data)+1).reshape(-1, 1), train_data.loc[:, 'y'])

        y_hat = model.predict(np.arange(1, points_to_predict+1).reshape(-1, 1))

        future_dataframe.iloc[-points_to_predict:, 0] = y_hat

        return future_dataframe
