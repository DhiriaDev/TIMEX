from pandas import DataFrame
import warnings
import itertools
import statsmodels.api as sm

from src.data_prediction.data_prediction import PredictionModel, post_transformation


class ARIMA(PredictionModel):
    """ARIMA prediction model."""
    # NOT WORKING

    def __init__(self, params: dict):
        super().__init__(params, "ARIMA")

    def train(self, input_data: DataFrame):
        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q))

        PERIOD_SAMPLES_FOR_DECOMPOSITION = 24
        START_TIME_INSTANT = 60
        SEASONAL_PLOT_HORIZON = 'day'
        # PERIOD_SAMPLES_FOR_DECOMPOSITION = 1440
        # START_TIME_INSTANT = 7000
        SUBSAMPLING_MODEL_EXPLORATION = ''
        # SUBSAMPLING_MODEL_EXPLORATION = 'H'

        if (SUBSAMPLING_MODEL_EXPLORATION != ''):
            if SUBSAMPLING_MODEL_EXPLORATION == 'H':
                PERIOD_SAMPLES_FOR_DECOMPOSITION = int(PERIOD_SAMPLES_FOR_DECOMPOSITION / 60)
                START_TIME_INSTANT = int(START_TIME_INSTANT / 60)
                resample = input_data.resample('H')
                input_data = resample.mean()

        seasonal_pdq = [(x[0], x[1], x[2], PERIOD_SAMPLES_FOR_DECOMPOSITION) for x in list(itertools.product(p, d, q))]

        min_AIC = 10e100

        for param in pdq:
            for param_seasonal in seasonal_pdq:
                # print(param)
                # print(param_seasonal)
                try:
                    mod = sm.tsa.statespace.SARIMAX(input_data.iloc[:, 0], order=param, seasonal_order=param_seasonal,
                                                    enforce_stationarity=False, enforce_invertibility=False)
                    results = mod.fit()
                    # print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

                    if results.aic < min_AIC:
                        min_AIC = results.aic
                        min_param = param
                        min_seasonal = param_seasonal
                except:
                    continue

            print('BEST SARIMA model:')
            print('ARIMA{}x{}12 - AIC:{}'.format(min_param, min_seasonal, min_AIC))

            print('FITTING the best SARIMA model')
            mod = sm.tsa.statespace.SARIMAX(input_data.iloc[:, 0], order=param, seasonal_order=param_seasonal,
                                            enforce_stationarity=False, enforce_invertibility=False)
            results = mod.fit()
            print(results.summary().tables[1])
            self.results = results



    def predict(self) -> DataFrame:
        """Overrides PredictionModel.predict()"""
        START_TIME_INSTANT = 60
        forecast = self.results.get_prediction(start=START_TIME_INSTANT, dynamic=False).prediction_results
        print(forecast)

        forecast.loc[:, 'yhat'] = post_transformation(forecast['yhat'], self.transformation)
        forecast.loc[:, 'yhat_lower'] = post_transformation(forecast['yhat_lower'], self.transformation)
        forecast.loc[:, 'yhat_upper'] = post_transformation(forecast['yhat_upper'], self.transformation)

        forecast.set_index('ds', inplace=True)

        return forecast
