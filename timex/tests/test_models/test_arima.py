import logging
import unittest

import numpy as np

from timex.data_prediction.arima_predictor import ARIMA
from timex.tests.utilities import get_fake_df


logger = logging.getLogger()
logger.level = logging.DEBUG

np.random.seed(0)


# stream_handler = logging.StreamHandler(sys.stdout)
# logger.addHandler(stream_handler)


class MyTestCase(unittest.TestCase):
    def test_launch_model_arima(self):
        param_config = {
            "model_parameters": {
                "test_percentage": 10,
                "delta_training_percentage": 20,
                "prediction_lags": 10,
                "transformation": "none",
                "main_accuracy_estimator": "mae"
            },
        }

        for n_threads in range(1, 5):
            param_config["max_threads"] = n_threads
            df = get_fake_df(100)
            predictor = ARIMA(param_config)
            model_result = predictor.launch_model(df.copy(), max_threads=n_threads)

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