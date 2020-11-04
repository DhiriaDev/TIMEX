import unittest

import pandas
from pandas import Series, DataFrame
import numpy as np

from src.data_ingestion.data_ingestion import data_ingestion
from src.data_prediction.arima_predictor import ARIMA
from src.data_prediction.data_prediction import pre_transformation
from src.data_prediction.prophet_predictor import FBProphet


class MyTestCase(unittest.TestCase):

    def test_pre_transformation_1(self):

        s = Series(np.array([1, 2, 3, 4]))
        res = pre_transformation(s, "log")

        self.assertEqual(res[0], np.log(1))
        self.assertEqual(res[1], np.log(2))
        self.assertEqual(res[2], np.log(3))
        self.assertEqual(res[3], np.log(4))

    def test_pre_transformation_2(self):

        s = Series(np.array([0]))
        res = pre_transformation(s, "log")

        self.assertEqual(res[0], 0)

    def test_pre_transformation_3(self):

        s = Series(np.array([2]))
        res = pre_transformation(s, "none")

        self.assertEqual(res[0], 2)

    def test_launch_model_fbprophet(self):
        param_config = {
            "verbose": "no",
            "model_parameters": {
                "test_percentage": 10,
                "delta_training_percentage": 20,
                "prediction_lags": 10,
                "transformation": "none",
                "main_accuracy_estimator": "mae"
            },
        }

        df = get_fake_df(100)
        predictor = FBProphet(param_config)
        model_result = predictor.launch_model(df.copy())

        self.assertEqual(predictor.test_values, 10)
        self.assertEqual(predictor.delta_training_values, 20)
        self.assertEqual(predictor.main_accuracy_estimator, "mae")

        self.assertEqual(len(model_result.results), 5)

        for r in model_result.results:
            prediction = r.prediction
            testing_performances = r.testing_performances
            first_used_index = testing_performances.first_used_index

            used_training_set = df.loc[first_used_index:]
            used_training_set = used_training_set.iloc[:-10]
            self.assertEqual(len(prediction), len(used_training_set) + 10 + 10)

    def test_launch_model_arima(self):
        param_config = {
            "verbose": "no",
            "input_parameters": {
                "source_data_url": "test_datasets/test_3.csv",
                "columns_to_load_from_url": "first_column,second_column",
                "datetime_column_name": "first_column",
                "index_column_name": "first_column",
                "datetime_format": "%Y-%m-%dT%H:%M:%S"
            },
            "model_parameters": {
                "test_percentage": 10,
                "delta_training_percentage": 90,
                "prediction_lags": 10,
                "transformation": "none",
                "main_accuracy_estimator": "mae"
            },
        }

        df = data_ingestion(param_config)
        predictor = ARIMA(param_config)
        arima_result = predictor.launch_model(df.copy())

        self.assertEqual(predictor.test_values, 1)
        self.assertEqual(predictor.delta_training_values, 3)
        self.assertGreater(len(arima_result.prediction), 10+3)


def get_fake_df(length: int) -> DataFrame:
    dates = pandas.date_range('1/1/2000', periods=length)

    np.random.seed(0)
    df = pandas.DataFrame(np.random.randn(length), index=dates, columns=['value'])
    return df


if __name__ == '__main__':
    unittest.main()