import unittest

from pandas import Series
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
        predictor = FBProphet(param_config)
        prophet_result = predictor.launch_model(df.copy())

        self.assertEqual(predictor.test_values, 1)
        self.assertEqual(predictor.delta_training_values, 9)
        self.assertEqual(len(prophet_result.prediction), 20)

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

if __name__ == '__main__':
    unittest.main()