import logging
import sys
import unittest

import pandas
from pandas import Series, DataFrame
import numpy as np
from scipy.stats import yeojohnson

from timex.data_prediction.arima_predictor import ARIMA
from timex.data_prediction.data_prediction import calc_xcorr
from timex.data_prediction.prophet_predictor import FBProphet
from timex.data_prediction.transformation import transformation_factory, Log
from timex.tests.utilities import get_fake_df

xcorr_modes = ['pearson', 'kendall', 'spearman', 'matlab_normalized']

logger = logging.getLogger()
logger.level = logging.DEBUG

np.random.seed(0)


# stream_handler = logging.StreamHandler(sys.stdout)
# logger.addHandler(stream_handler)


class MyTestCase(unittest.TestCase):
    def test_transformation_log(self):
        s = Series(np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]))
        tr = transformation_factory("log")
        res = tr.apply(s)

        self.assertEqual(res[0], -np.log(4))
        self.assertEqual(res[1], -np.log(3))
        self.assertEqual(res[2], -np.log(2))
        self.assertEqual(res[3], -np.log(1))
        self.assertEqual(res[4],  0)
        self.assertEqual(res[5],  np.log(1))
        self.assertEqual(res[6],  np.log(2))
        self.assertEqual(res[7],  np.log(3))
        self.assertEqual(res[8],  np.log(4))

        res = tr.inverse(res)

        exp = Series(np.array([-4, -3, -2, 0, 0, 0, 2, 3, 4]))
        self.assertTrue(np.allclose(res, exp))
        self.assertEqual(str(tr), "Log")

    def test_transformation_log_modified(self):
        s = Series(np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]))
        tr = transformation_factory("log_modified")
        res = tr.apply(s)

        self.assertEqual(res[0], -np.log(4+1))
        self.assertEqual(res[1], -np.log(3+1))
        self.assertEqual(res[2], -np.log(2+1))
        self.assertEqual(res[3], -np.log(1+1))
        self.assertEqual(res[4],  np.log(1))
        self.assertEqual(res[5],  np.log(1+1))
        self.assertEqual(res[6],  np.log(2+1))
        self.assertEqual(res[7],  np.log(3+1))
        self.assertEqual(res[8],  np.log(4+1))

        res = tr.inverse(res)

        self.assertTrue(np.allclose(s, res))
        self.assertEqual(str(tr), "modified Log")

    def test_transformation_yeo_johnson(self):
        s = Series(np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]))
        tr = transformation_factory("yeo_johnson")
        res = tr.apply(s)
        lmbda = tr.lmbda

        exp = yeojohnson(s, lmbda)

        self.assertTrue(np.allclose(res, exp))

        res = tr.inverse(res)

        self.assertTrue(np.allclose(s, res))
        self.assertEqual(str(tr), f"Yeo-Johnson (lambda: {round(lmbda, 3)})")

    # def test_transformation_yeo_johnson_2(self):
    #     a = Series(np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]))
    #     b = Series(np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]))
    #     c = Series(np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]))
    #
    #     df = DataFrame(data={"a": a, "b": b, "c": c})
    #     tr = transformation_factory("yeo_johnson")
    #
    #     res_a = tr.apply(df["a"])
    #     res_b = tr.apply(df["b"])
    #     res_c = tr.apply(df["c"])
    #
    #     print(res_a)
    #
    #     res_a = tr.inverse(res_a)
    #     res_b = tr.inverse(res_b)
    #     res_c = tr.inverse(res_c)
    #
    #     self.assertTrue(np.allclose(a, res_a))
    #     self.assertTrue(np.allclose(b, res_b))
    #     self.assertTrue(np.allclose(c, res_c))
    #
    #     lmbda = tr.lmbda["a"]
    #     self.assertEqual(str(tr), f"Yeo-Johnson (lambda: {round(lmbda)})")

    def test_transformation_diff(self):
        s = Series(np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]))
        tr = transformation_factory("diff")

        res = tr.apply(s)

        self.assertEqual(res[1],   1.0)
        self.assertEqual(res[2],   1.0)
        self.assertEqual(res[3],   1.0)
        self.assertEqual(res[4],   1.0)
        self.assertEqual(res[5],   1.0)
        self.assertEqual(res[6],   1.0)
        self.assertEqual(res[7],   1.0)
        self.assertEqual(res[8],   1.0)
        self.assertEqual(tr.first_value, -4)

        res = tr.inverse(res)

        self.assertTrue(np.allclose(res, s))

        self.assertEqual(str(tr), f"differentiate (1)")

    def test_transformation_none(self):
        s = Series(np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]))
        tr = transformation_factory("none")
        res = tr.apply(s)

        self.assertEqual(res[0], -4)
        self.assertEqual(res[1], -3)
        self.assertEqual(res[2], -2)
        self.assertEqual(res[3], -1)
        self.assertEqual(res[4],  0)
        self.assertEqual(res[5],  1)
        self.assertEqual(res[6],  2)
        self.assertEqual(res[7],  3)
        self.assertEqual(res[8],  4)

        res = tr.inverse(res)

        self.assertTrue(np.allclose(s, res))

    def test_launch_model_fbprophet_1(self):
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

        for n_threads in range(1, 7):
            param_config["max_threads"] = n_threads
            df = get_fake_df(100)
            predictor = FBProphet(param_config)
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

    def test_launch_model_fbprophet_1_1(self):
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

        for n_threads in range(1, 7):
            param_config["max_threads"] = n_threads
            df = get_fake_df(101)
            predictor = FBProphet(param_config)
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

    def test_launch_model_fbprophet_2(self):
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

        for n_threads in range(1, 7):
            df = get_fake_df(100)
            predictor = FBProphet(param_config)
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

    def test_launch_model_fbprophet_3(self):
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
            predictor = FBProphet(param_config)
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

    def test_launch_model_fbprophet_4(self):
        # Check default parameters.
        param_config = {
            "verbose": "no",
        }

        for n_threads in range(1, 7):
            param_config["max_threads"] = n_threads
            df = get_fake_df(10)
            predictor = FBProphet(param_config)
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

    def test_launch_model_fbprophet_5(self):
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

        for n_threads in range(1, 7):
            param_config["max_threads"] = n_threads
            df = get_fake_df(10)
            extra_regressor = get_fake_df(20, name="A")

            predictor = FBProphet(param_config)
            model_result = predictor.launch_model(df.copy(), extra_regressors=extra_regressor)

            expected_train_set_lengths = [2, 4, 6, 8, 9]

            self.assertEqual(predictor.test_values, 1)
            self.assertEqual(predictor.delta_training_values, 2)
            self.assertEqual(predictor.main_accuracy_estimator, "mae")

            self.assertEqual(len(model_result.results), 5)

            i = 0
            for r in model_result.results:
                prediction = r.prediction
                testing_performances = r.testing_performances
                first_used_index = testing_performances.first_used_index

                used_training_set = df.loc[first_used_index:]
                used_training_set = used_training_set.iloc[:-1]
                self.assertEqual(len(prediction), len(used_training_set) + 1 + 10)
                self.assertEqual(len(used_training_set), expected_train_set_lengths[i])
                i += 1

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

    def test_calc_xcorr_1(self):
        # Example from https://www.mathworks.com/help/matlab/ref/xcorr.html, slightly modified
        # The y series is antecedent the x series; therefore, a correlation should be found between x and y.

        x = [np.power(0.94, n) for n in np.arange(0, 100)]
        noise = np.random.normal(0, 1, 100)
        x = x + noise

        y = np.roll(x, -5)
        z = np.roll(x, -10)

        df = DataFrame(data={"x": x, "y": y, "z": z})
        df = df.iloc[0:-10]

        xcorr = calc_xcorr("x", df, max_lags=10, modes=xcorr_modes)
        for mode in xcorr:
            self.assertEqual(xcorr[mode].idxmax()[0], 5)
            self.assertEqual(xcorr[mode].idxmax()[1], 10)

    def test_calc_xcorr_2(self):
        # Shift a sin. Verify that highest correlation is in the correct region.

        # Restrict the delay to less than one period of the sin.
        max_delay = 50 - 1
        x = np.linspace(0, 2 * np.pi, 200)
        y = np.sin(x)

        noise = np.random.normal(0, 0.5, 200)
        y = y + noise

        for i in range(-max_delay, max_delay):
            y_delayed = np.roll(y, i)

            df = DataFrame(data={"y": y, "y_delayed": y_delayed})

            xcorr = calc_xcorr("y", df, max_lags=max_delay, modes=xcorr_modes)
            expected_max_lag = -i
            for mode in xcorr:
                self.assertLess(abs(xcorr[mode].idxmax()[0] - expected_max_lag), 4)

    def test_calc_xcorr_granger(self):
        # Shift a sin. Verify that highest correlation is in the correct region.
        # Specific test for granger method, which is slightly different from the others.

        # Restrict the delay to less than one period of the sin.
        max_delay = 50 - 1
        x = np.linspace(0, 2 * np.pi, 200)
        y = np.sin(x)

        noise = np.random.normal(0, 1, 200)
        y = y + noise

        for i in [x for x in range(-max_delay, max_delay) if x != 0]:
            y_delayed = np.roll(y, i)

            df = DataFrame(data={"y": y, "y_delayed": y_delayed})

            xcorr = calc_xcorr("y", df, max_lags=max_delay, modes=['granger'])
            expected_max_lag = -i
            for mode in xcorr:
                self.assertEqual(xcorr[mode].loc[expected_max_lag][0], 1.0)


if __name__ == '__main__':
    unittest.main()