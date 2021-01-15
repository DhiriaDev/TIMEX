import logging
import unittest

import numpy as np

from timex.data_prediction.lstm_predictor import LSTM_model
from timex.data_prediction.transformation import Log
from timex.tests.utilities import get_fake_df

logger = logging.getLogger()
logger.level = logging.DEBUG

np.random.seed(0)
max_threads_to_test = 1

# stream_handler = logging.StreamHandler(sys.stdout)
# logger.addHandler(stream_handler)


class MyTestCase(unittest.TestCase):

    def test_launch_model_lstm_1(self):
        # Percentages' sum is not 100%; adapt the windows. Use "test_percentage".
        param_config = {
            "model_parameters": {
                "test_percentage": 10,
                "delta_training_percentage": 20,
                "prediction_lags": 10,
                "transformation": "none",
                "main_accuracy_estimator": "mae"
            },
        }

        for n_threads in range(1, max_threads_to_test):
            param_config["max_threads"] = n_threads
            df = get_fake_df(100)
            predictor = LSTM_model(param_config)
            model_result = predictor.launch_model(df.copy(), max_threads=n_threads)

            expected_train_set_lengths = [20, 40, 60, 80, 90]

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
                expected_train_set_lengths.remove(len(used_training_set))

            self.assertEqual(len(expected_train_set_lengths), 0)

    def test_launch_model_lstm_1_1(self):
        # Percentages' sum is not 100% and values are float; adapt the windows. Use "test_percentage".
        param_config = {
            "model_parameters": {
                "test_percentage": 11.7,
                "delta_training_percentage": 18.9,
                "prediction_lags": 10,
                "transformation": "none",
                "main_accuracy_estimator": "mae"
            },
        }

        for n_threads in range(1, max_threads_to_test):
            param_config["max_threads"] = n_threads
            df = get_fake_df(101)
            predictor = LSTM_model(param_config)
            model_result = predictor.launch_model(df.copy(), max_threads=n_threads)

            expected_train_set_lengths = [19, 38, 57, 76, 89]

            self.assertEqual(predictor.test_values, 12)
            self.assertEqual(predictor.delta_training_values, 19)
            self.assertEqual(predictor.main_accuracy_estimator, "mae")

            self.assertEqual(len(model_result.results), 5)

            for r in model_result.results:
                prediction = r.prediction
                testing_performances = r.testing_performances
                first_used_index = testing_performances.first_used_index

                used_training_set = df.loc[first_used_index:]
                used_training_set = used_training_set.iloc[:-12]
                self.assertEqual(len(prediction), len(used_training_set) + 10 + 12)
                expected_train_set_lengths.remove(len(used_training_set))

            self.assertEqual(len(expected_train_set_lengths), 0)

    def test_launch_model_lstm_2(self):
        # Percentages' sum is not 100%; adapt the windows. Use "test_values".
        param_config = {
            "model_parameters": {
                "test_values": 5,
                "delta_training_percentage": 20,
                "prediction_lags": 10,
                "transformation": "none",
                "main_accuracy_estimator": "mae"
            },
        }

        for n_threads in range(1, max_threads_to_test):
            df = get_fake_df(100)
            predictor = LSTM_model(param_config)
            model_result = predictor.launch_model(df.copy(), max_threads=n_threads)

            expected_train_set_lengths = [20, 40, 60, 80, 95]

            self.assertEqual(predictor.test_values, 5)
            self.assertEqual(predictor.delta_training_values, 20)
            self.assertEqual(predictor.main_accuracy_estimator, "mae")

            self.assertEqual(len(model_result.results), 5)

            for r in model_result.results:
                prediction = r.prediction
                testing_performances = r.testing_performances
                first_used_index = testing_performances.first_used_index

                used_training_set = df.loc[first_used_index:]
                used_training_set = used_training_set.iloc[:-5]
                self.assertEqual(len(prediction), len(used_training_set) + 5 + 10)
                expected_train_set_lengths.remove(len(used_training_set))

            self.assertEqual(len(expected_train_set_lengths), 0)

    def test_launch_model_lstm_3(self):
        # Percentages' sum is over 100%; adapt the window.
        param_config = {
            "model_parameters": {
                "test_values": 5,
                "delta_training_percentage": 100,
                "prediction_lags": 10,
                "transformation": "none",
                "main_accuracy_estimator": "mae"
            },
        }

        for n_threads in range(1, 3):
            param_config["max_threads"] = n_threads
            df = get_fake_df(100)
            predictor = LSTM_model(param_config)
            model_result = predictor.launch_model(df.copy(), max_threads=n_threads)

            expected_train_set_lengths = [95]

            self.assertEqual(predictor.test_values, 5)
            self.assertEqual(predictor.delta_training_values, 100)
            self.assertEqual(predictor.main_accuracy_estimator, "mae")

            self.assertEqual(len(model_result.results), 1)

            for r in model_result.results:
                prediction = r.prediction
                testing_performances = r.testing_performances
                first_used_index = testing_performances.first_used_index

                used_training_set = df.loc[first_used_index:]
                used_training_set = used_training_set.iloc[:-5]
                self.assertEqual(len(prediction), len(used_training_set) + 5 + 10)
                expected_train_set_lengths.remove(len(used_training_set))

            self.assertEqual(len(expected_train_set_lengths), 0)

    def test_launch_model_lstm_4(self):
        # Check default parameters.
        param_config = {
            "verbose": "no",
        }

        for n_threads in range(1, max_threads_to_test):
            param_config["max_threads"] = n_threads
            df = get_fake_df(10)
            predictor = LSTM_model(param_config)
            model_result = predictor.launch_model(df.copy(), max_threads=n_threads)

            expected_train_set_lengths = [2, 4, 6, 8, 9]

            self.assertEqual(predictor.test_values, 1)
            self.assertEqual(predictor.delta_training_values, 2)
            self.assertEqual(predictor.main_accuracy_estimator, "mae")
            self.assertEqual(type(predictor.transformation), Log)

            self.assertEqual(len(model_result.results), 5)

            for r in model_result.results:
                prediction = r.prediction
                testing_performances = r.testing_performances
                first_used_index = testing_performances.first_used_index

                used_training_set = df.loc[first_used_index:]
                used_training_set = used_training_set.iloc[:-1]
                self.assertEqual(len(prediction), len(used_training_set) + 1 + 10)
                expected_train_set_lengths.remove(len(used_training_set))

            self.assertEqual(len(expected_train_set_lengths), 0)

    def test_launch_model_lstm_5(self):
        # Test extra-regressors.
        param_config = {
            "model_parameters": {
                "test_percentage": 10,
                "delta_training_percentage": 20,
                "prediction_lags": 10,
                "transformation": "none",
                "main_accuracy_estimator": "mae"
            },
        }

        for n_threads in range(1, max_threads_to_test):
            param_config["max_threads"] = n_threads
            df = get_fake_df(10)
            extra_regressor = get_fake_df(20, name="A")

            predictor = LSTM_model(param_config)
            model_result = predictor.launch_model(df.copy(), extra_regressors=extra_regressor)

            expected_train_set_lengths = [2, 4, 6, 8, 9]

            self.assertEqual(predictor.test_values, 1)
            self.assertEqual(predictor.delta_training_values, 2)
            self.assertEqual(predictor.main_accuracy_estimator, "mae")
            self.assertEqual(len(model_result.results), 5)

            for r in model_result.results:
                prediction = r.prediction
                testing_performances = r.testing_performances
                first_used_index = testing_performances.first_used_index

                used_training_set = df.loc[first_used_index:]
                used_training_set = used_training_set.iloc[:-1]
                self.assertEqual(len(prediction), len(used_training_set) + 1 + 10)
                expected_train_set_lengths.remove(len(used_training_set))

            self.assertEqual(len(expected_train_set_lengths), 0)


if __name__ == '__main__':
    unittest.main()