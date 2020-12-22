import os
import unittest
from datetime import datetime

import pandas
from pandas import DataFrame
import numpy as np

from timex.data_ingestion.data_ingestion import add_freq
from timex.data_prediction.data_prediction import SingleResult, TestingPerformance, ModelResult, calc_all_xcorr
from timex.scenario.scenario import Scenario
from timex.utils.utils import prepare_extra_regressor, get_best_univariate_predictions, \
    get_best_multivariate_predictions, compute_predictions


class MyTestCase(unittest.TestCase):

    def test_prepare_extra_regressors(self):
        ing_data = DataFrame({"a": np.arange(0, 10), "b": np.arange(10, 20)})
        ing_data.set_index("a", inplace=True)

        forecast = DataFrame({"a": np.arange(8, 15), "yhat": np.arange(40, 47)})
        forecast.set_index("a", inplace=True)

        tp = TestingPerformance(first_used_index=0)
        tp.MAE = 0

        model_results = [SingleResult(forecast, tp)]
        models = {'fbprophet': ModelResult(model_results, None, None)}
        scenario = Scenario(ing_data, models, None)

        result = prepare_extra_regressor(scenario, 'fbprophet', 'MAE')

        expected = DataFrame({"a": np.arange(0, 15),
                              "b": np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 42.0, 43.0,
                                             44.0, 45.0, 46.0])})
        expected.set_index("a", inplace=True)

        self.assertTrue(expected.equals(result))

    def test_get_best_univariate_and_multivariate_predictions(self):
        # Check that results are in the correct form.

        ing_data = DataFrame({"a": pandas.date_range('1/1/2000', periods=30),
                              "b": np.arange(30, 60), "c": np.arange(60, 90)})
        ing_data.set_index("a", inplace=True)
        ing_data = add_freq(ing_data, "D")

        param_config = {
            "model_parameters": {
                "test_values": 2,
                "delta_training_percentage": 20,
                "prediction_lags": 10,
                "possible_transformations": "log_modified,none",
                "models": "fbprophet,arima",
                "main_accuracy_estimator": "mae",
                "xcorr_max_lags": 120,
                "xcorr_extra_regressor_threshold": 0.8,
                "xcorr_mode": "pearson",
                "xcorr_mode_target": "pearson"
            }
        }

        total_xcorr = calc_all_xcorr(ingested_data=ing_data, param_config=param_config)

        best_transformations, scenarios = get_best_univariate_predictions(ing_data, param_config, total_xcorr)

        self.assertTrue(len(best_transformations), 2)
        self.assertTrue(best_transformations["fbprophet"]["b"] in ["log_modified", "none"])
        self.assertTrue(best_transformations["fbprophet"]["c"] in ["log_modified", "none"])
        self.assertTrue(best_transformations["arima"]["b"] in ["log_modified", "none"])
        self.assertTrue(best_transformations["arima"]["c"] in ["log_modified", "none"])

        self.assertEqual(len(scenarios), 2)
        self.assertEqual(scenarios[0].scenario_data.columns[0], "b")
        self.assertEqual(scenarios[1].scenario_data.columns[0], "c")

        self.assertEqual(len(scenarios[0].models), 2)
        self.assertEqual(len(scenarios[1].models), 2)

        scenarios = get_best_multivariate_predictions(best_transformations=best_transformations, ingested_data=ing_data,
                                                      scenarios=scenarios, param_config=param_config,
                                                      total_xcorr=total_xcorr)
        self.assertEqual(len(scenarios), 2)
        self.assertEqual(scenarios[0].scenario_data.columns[0], "b")
        self.assertEqual(scenarios[1].scenario_data.columns[0], "c")

        self.assertEqual(len(scenarios[0].models), 2)
        self.assertEqual(len(scenarios[1].models), 2)

    def test_compute_predictions(self):
        # Check results are in the correct form and test the function to save historic predictions to file.
        ing_data = DataFrame({"a": pandas.date_range('2000-01-01', periods=30),
                              "b": np.arange(30, 60), "c": np.arange(60, 90)})
        ing_data.set_index("a", inplace=True)
        ing_data = add_freq(ing_data, "D")

        param_config = {
            "input_parameters": {},
            "model_parameters": {
                "test_values": 2,
                "delta_training_percentage": 20,
                "prediction_lags": 10,
                "possible_transformations": "log_modified,none",
                "models": "fbprophet,arima",
                "main_accuracy_estimator": "mae",
                "xcorr_max_lags": 120,
                "xcorr_extra_regressor_threshold": 0.8,
                "xcorr_mode": "pearson",
                "xcorr_mode_target": "pearson"
            },
            "historical_prediction_parameters": {
                "initial_index": "2000-01-29",
                "save_path": "test_hist_pred_saves/test1.pkl"
            }
        }

        scenarios = compute_predictions(ingested_data=ing_data, param_config=param_config)

        self.assertEqual(len(scenarios), 2)
        self.assertEqual(scenarios[0].scenario_data.columns[0], "b")
        self.assertEqual(scenarios[1].scenario_data.columns[0], "c")

        self.assertEqual(len(scenarios[0].models), 2)
        self.assertEqual(len(scenarios[1].models), 2)

        b_old_hist = scenarios[0].historical_prediction
        c_old_hist = scenarios[1].historical_prediction

        for s in scenarios:
            for model in s.historical_prediction:
                hist_prediction = s.historical_prediction[model]
                self.assertEqual(len(hist_prediction), 2)
                self.assertEqual(hist_prediction.index[0], pandas.to_datetime('2000-01-30', format="%Y-%m-%d"))
                self.assertEqual(hist_prediction.index[1], pandas.to_datetime('2000-01-31', format="%Y-%m-%d"))

        # Simulate a 1-step ahead in time, so we have collected a new point.
        # Note that past values are changed as well, so we will check that TIMEX does not change the old predictions.
        ing_data = DataFrame({"a": pandas.date_range('2000-01-01', periods=31),
                              "b": np.arange(20, 51), "c": np.arange(35, 66)})
        ing_data.set_index("a", inplace=True)
        ing_data = add_freq(ing_data, "D")

        # This time historical predictions will be loaded from file.
        scenarios = compute_predictions(ingested_data=ing_data, param_config=param_config)

        for s in scenarios:
            for model in s.historical_prediction:
                hist_prediction = s.historical_prediction[model]
                self.assertEqual(len(hist_prediction), 3)
                self.assertEqual(hist_prediction.index[0], pandas.to_datetime('2000-01-30', format="%Y-%m-%d"))
                self.assertEqual(hist_prediction.index[1], pandas.to_datetime('2000-01-31', format="%Y-%m-%d"))
                self.assertEqual(hist_prediction.index[2], pandas.to_datetime('2000-02-01', format="%Y-%m-%d"))

        # Check that past predictions have not been touched.
        self.assertEqual(b_old_hist['fbprophet'].iloc[0, 0], scenarios[0].historical_prediction['fbprophet'].iloc[0, 0])
        self.assertEqual(b_old_hist['fbprophet'].iloc[1, 0], scenarios[0].historical_prediction['fbprophet'].iloc[1, 0])
        self.assertEqual(b_old_hist['arima'].iloc[0, 0], scenarios[0].historical_prediction['arima'].iloc[0, 0])
        self.assertEqual(b_old_hist['arima'].iloc[1, 0], scenarios[0].historical_prediction['arima'].iloc[1, 0])

        self.assertEqual(c_old_hist['fbprophet'].iloc[0, 0], scenarios[1].historical_prediction['fbprophet'].iloc[0, 0])
        self.assertEqual(c_old_hist['fbprophet'].iloc[1, 0], scenarios[1].historical_prediction['fbprophet'].iloc[1, 0])
        self.assertEqual(c_old_hist['arima'].iloc[0, 0], scenarios[1].historical_prediction['arima'].iloc[0, 0])
        self.assertEqual(c_old_hist['arima'].iloc[1, 0], scenarios[1].historical_prediction['arima'].iloc[1, 0])

        # Cleanup.
        os.remove("test_hist_pred_saves/test1.pkl")


if __name__ == '__main__':
    unittest.main()