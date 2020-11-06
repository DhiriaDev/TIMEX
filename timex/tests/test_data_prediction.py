import unittest

from pandas import Series
import numpy as np

from timex.data_prediction.arima_predictor import ARIMA
from timex.data_prediction.data_prediction import pre_transformation
from timex.data_prediction.prophet_predictor import FBProphet
from timex.tests.utilities import get_fake_df


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
            "model_parameters": {
                "test_percentage": 10,
                "delta_training_percentage": 20,
                "prediction_lags": 10,
                "transformation": "none",
                "main_accuracy_estimator": "mae"
            },
        }

        df = get_fake_df(100)
        predictor = ARIMA(param_config)
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


if __name__ == '__main__':
    unittest.main()