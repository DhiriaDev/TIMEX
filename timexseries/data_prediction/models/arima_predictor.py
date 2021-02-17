from pandas import DataFrame
import warnings
import itertools
import statsmodels.api as sm
import pandas as pd

from timexseries.data_prediction import PredictionModel


class ARIMAModel(PredictionModel):
    """ARIMA prediction model."""

    # NOT WORKING

    def __init__(self, params: dict, transformation: str = None):
        super().__init__(params, name="ARIMA", transformation=transformation)

    def train(self, input_data: DataFrame, extra_regressor: DataFrame = None):
        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

        results = []
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        mod = sm.tsa.statespace.SARIMAX(input_data, order=param, seasonal_order=param_seasonal,
                                                        enforce_stationarity=False, enforce_invertibility=False)
                        result = mod.fit(disp=0)
                        results.append((param, param_seasonal, result.aic))
                except:
                    continue

        results.sort(key=lambda x: x[2])

        mod = sm.tsa.statespace.SARIMAX(input_data,
                                        order=results[0][0],
                                        seasonal_order=results[0][1],
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        # TODO: avoid re-training again
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = mod.fit(disp=0)

    def predict(self, future_dataframe: DataFrame, extra_regressor: DataFrame = None) -> DataFrame:
        """Overrides PredictionModel.predict()"""
        pred = self.model.forecast(future_dataframe.index.values[-1])

        r = pd.DataFrame(pred)
        r.rename(columns={'predicted_mean': 'yhat'}, inplace=True)
        future_dataframe.update(r)

        return future_dataframe
