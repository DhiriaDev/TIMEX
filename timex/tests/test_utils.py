import os

import dateparser
import pandas
import pytest
from fbprophet import Prophet
from pandas import DataFrame
import numpy as np
import pandas as pd

from timex.data_ingestion.data_ingestion import add_freq
from timex.data_prediction.data_prediction import SingleResult, TestingPerformance, ModelResult, calc_all_xcorr
from timex.data_prediction.prophet_predictor import suppress_stdout_stderr
from timex.scenario.scenario import Scenario
from timex.utils.utils import prepare_extra_regressor, get_best_univariate_predictions, \
    get_best_multivariate_predictions, compute_historical_predictions, create_scenarios


class TestGetPredictions:

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

        assert expected.equals(result)

    def test_get_best_univariate_and_multivariate_predictions(self):
        # Check that results are in the correct form.

        ing_data = DataFrame({"a": pandas.date_range('1/1/2000', periods=30),
                              "b": np.arange(30, 60), "c": np.arange(60, 90)})
        ing_data.set_index("a", inplace=True)
        ing_data = add_freq(ing_data, "D")

        param_config = {
            "xcorr_parameters": {
                "xcorr_max_lags": 120,
                "xcorr_extra_regressor_threshold": 0.8,
                "xcorr_mode": "pearson",
                "xcorr_mode_target": "pearson"
            },
            "model_parameters": {
                "test_values": 2,
                "delta_training_percentage": 20,
                "prediction_lags": 10,
                "possible_transformations": "log_modified,none",
                "models": "fbprophet,mockup",
                "main_accuracy_estimator": "mae",
            }
        }

        total_xcorr = calc_all_xcorr(ingested_data=ing_data, param_config=param_config)

        best_transformations, scenarios = get_best_univariate_predictions(ing_data, param_config, total_xcorr)

        assert len(best_transformations) == 2
        assert best_transformations["fbprophet"]["b"] in ["log_modified", "none"]
        assert best_transformations["fbprophet"]["c"] in ["log_modified", "none"]
        assert best_transformations["mockup"]["b"] in ["log_modified", "none"]
        assert best_transformations["mockup"]["c"] in ["log_modified", "none"]

        assert len(scenarios) == 2
        assert scenarios[0].scenario_data.columns[0] == "b"
        assert scenarios[1].scenario_data.columns[0] == "c"

        assert len(scenarios[0].models) == 2
        assert len(scenarios[1].models) == 2

        scenarios = get_best_multivariate_predictions(best_transformations=best_transformations, ingested_data=ing_data,
                                                      scenarios=scenarios, param_config=param_config,
                                                      total_xcorr=total_xcorr)
        assert len(scenarios) == 2
        assert scenarios[0].scenario_data.columns[0] == "b"
        assert scenarios[1].scenario_data.columns[0] == "c"

        assert len(scenarios[0].models) == 2
        assert len(scenarios[1].models) == 2

    def test_compute_predictions(self):
        # Check results are in the correct form and test the function to save historic predictions to file.
        ing_data = DataFrame({"a": pandas.date_range('2000-01-01', periods=30),
                              "b": np.arange(30, 60), "c": np.arange(60, 90)})
        ing_data.set_index("a", inplace=True)
        ing_data = add_freq(ing_data, "D")

        param_config = {
            "xcorr_parameters": {
                "xcorr_max_lags": 120,
                "xcorr_extra_regressor_threshold": 0.8,
                "xcorr_mode": "pearson",
                "xcorr_mode_target": "pearson"
            },
            "input_parameters": {},
            "model_parameters": {
                "test_values": 2,
                "delta_training_percentage": 20,
                "prediction_lags": 10,
                "possible_transformations": "log_modified,none",
                "models": "fbprophet,mockup",
                "main_accuracy_estimator": "mae",
            },
            "historical_prediction_parameters": {
                "initial_index": "2000-01-29",
                "save_path": "test_hist_pred_saves/test1.pkl"
            }
        }

        # Cleanup eventual dumps.
        try:
            os.remove("test_hist_pred_saves/test1.pkl")
        except FileNotFoundError:
            pass

        scenarios = compute_historical_predictions(ingested_data=ing_data, param_config=param_config)

        assert len(scenarios) == 2
        assert scenarios[0].scenario_data.columns[0] == "b"
        assert scenarios[1].scenario_data.columns[0] == "c"

        assert len(scenarios[0].models) == 2
        assert len(scenarios[1].models) == 2

        b_old_hist = scenarios[0].historical_prediction
        c_old_hist = scenarios[1].historical_prediction

        for s in scenarios:
            for model in s.historical_prediction:
                hist_prediction = s.historical_prediction[model]
                assert len(hist_prediction) == 2
                assert hist_prediction.index[0] == pandas.to_datetime('2000-01-30', format="%Y-%m-%d")
                assert hist_prediction.index[1] == pandas.to_datetime('2000-01-31', format="%Y-%m-%d")

        # Simulate a 1-step ahead in time, so we have collected a new point.
        # Note that past values are changed as well, so we will check that TIMEX does not change the old predictions.
        ing_data = DataFrame({"a": pandas.date_range('2000-01-01', periods=31),
                              "b": np.arange(20, 51), "c": np.arange(35, 66)})
        ing_data.set_index("a", inplace=True)
        ing_data = add_freq(ing_data, "D")

        # This time historical predictions will be loaded from file.
        scenarios = compute_historical_predictions(ingested_data=ing_data, param_config=param_config)

        for s in scenarios:
            for model in s.historical_prediction:
                hist_prediction = s.historical_prediction[model]
                assert len(hist_prediction) == 3
                assert hist_prediction.index[0] == pandas.to_datetime('2000-01-30', format="%Y-%m-%d")
                assert hist_prediction.index[1] == pandas.to_datetime('2000-01-31', format="%Y-%m-%d")
                assert hist_prediction.index[2] == pandas.to_datetime('2000-02-01', format="%Y-%m-%d")

        # Check that past predictions have not been touched.
        assert b_old_hist['fbprophet'].iloc[0, 0] == scenarios[0].historical_prediction['fbprophet'].iloc[0, 0]
        assert b_old_hist['fbprophet'].iloc[1, 0] == scenarios[0].historical_prediction['fbprophet'].iloc[1, 0]
        assert b_old_hist['mockup'].iloc[0, 0] == scenarios[0].historical_prediction['mockup'].iloc[0, 0]
        assert b_old_hist['mockup'].iloc[1, 0] == scenarios[0].historical_prediction['mockup'].iloc[1, 0]

        assert c_old_hist['fbprophet'].iloc[0, 0] == scenarios[1].historical_prediction['fbprophet'].iloc[0, 0]
        assert c_old_hist['fbprophet'].iloc[1, 0] == scenarios[1].historical_prediction['fbprophet'].iloc[1, 0]
        assert c_old_hist['mockup'].iloc[0, 0] == scenarios[1].historical_prediction['mockup'].iloc[0, 0]
        assert c_old_hist['mockup'].iloc[1, 0] == scenarios[1].historical_prediction['mockup'].iloc[1, 0]

        # Cleanup.
        os.remove("test_hist_pred_saves/test1.pkl")

    def test_compute_predictions_2(self):

        ing_data = pd.read_csv("test_datasets/test_covid.csv")
        ing_data["data"] = ing_data["data"].apply(lambda x: dateparser.parse(x))
        ing_data.set_index("data", inplace=True, drop=True)
        ing_data = add_freq(ing_data, "D")

        param_config = {
            "input_parameters": {},
            "model_parameters": {
                "test_values": 5,
                "delta_training_percentage": 30,
                "prediction_lags": 10,
                "possible_transformations": "none",
                "models": "fbprophet",
                "main_accuracy_estimator": "mae",
            },
            "historical_prediction_parameters": {
                "initial_index": "2020-12-08",
                "save_path": "test_hist_pred_saves/test2.pkl"
            }
        }

        # You can verify with this code that tr_1 is the best training window.
        # test_values = 5
        # tr_1 = ing_data.copy().iloc[-35:-5][['nuovi_positivi']]
        # tr_2 = ing_data.copy().iloc[-65:-5][['nuovi_positivi']]
        # tr_3 = ing_data.copy().iloc[-95:-5][['nuovi_positivi']]
        # tr_4 = ing_data.copy().iloc[0:-5][['nuovi_positivi']]

        # tr_sets = [tr_1, tr_2, tr_3, tr_4]
        # testing_df = ing_data.copy().iloc[-5:]['nuovi_positivi']
        #
        # for tr in tr_sets:
        #     fb_tr = tr.copy()
        #     fbmodel = Prophet()
        #     fb_tr.reset_index(inplace=True)
        #     fb_tr.columns = ['ds', 'y']
        #
        #     with suppress_stdout_stderr():
        #         fbmodel.fit(fb_tr)
        #
        #     future_df = pd.DataFrame(index=pd.date_range(freq="1d",
        #                                                  start=tr.index.values[0],
        #                                                  periods=len(tr) + test_values + 10),
        #                              columns=["yhat"], dtype=tr.iloc[:, 0].dtype)
        #
        #     future = future_df.reset_index()
        #     future.rename(columns={'index': 'ds'}, inplace=True)
        #
        #     forecast = fbmodel.predict(future)
        #
        #     forecast.set_index('ds', inplace=True)
        #
        #     testing_prediction = forecast.iloc[-15:-10]['yhat']
        #     print(mean_absolute_error(testing_df['nuovi_positivi'], testing_prediction))

        # The best tr is tr_1. Compute historical predictions.
        tr_1 = ing_data.copy().iloc[-35:][['nuovi_positivi']]
        fb_tr = tr_1.copy()
        fbmodel = Prophet()
        fb_tr.reset_index(inplace=True)
        fb_tr.columns = ['ds', 'y']

        with suppress_stdout_stderr():
            fbmodel.fit(fb_tr)

        future_df = pd.DataFrame(index=pd.date_range(freq="1d",
                                                     start=tr_1.index.values[0],
                                                     periods=len(tr_1) + 10),
                                 columns=["yhat"], dtype=tr_1.iloc[:, 0].dtype)
        future = future_df.reset_index()
        future.rename(columns={'index': 'ds'}, inplace=True)
        forecast = fbmodel.predict(future)
        forecast.set_index('ds', inplace=True)
        historical_prediction = forecast[['yhat']]

        # Let TIMEX do this thing.
        scenarios = compute_historical_predictions(ingested_data=ing_data, param_config=param_config)

        scenario = scenarios[1]
        training_results = scenario.models['fbprophet'].results
        training_results.sort(key=lambda x: getattr(x.testing_performances, 'MAE'))

        assert historical_prediction.equals(scenario.models['fbprophet'].best_prediction[['yhat']])

        # Cleanup.
        os.remove("test_hist_pred_saves/test2.pkl")

        # Make this test with a log_modified


class TestCreateScenarios:
    @pytest.mark.parametrize(
        "historical_predictions, xcorr, additional_regressors, expected_extra_regressors, expected_value",
        [(True,  True,  True,  {"b": "c, d", "c": "b, e"}, 2.0),
         (True,  True,  False, {"b": "c", "c": "b"},       1.0),
         (True,  False, True,  {"b": "d", "c": "e"},       1.0),
         (True,  False, False, {},                         0.0),
         (False, True,  True,  {"b": "c, d", "c": "b, e"}, 2.0),
         (False, True,  False, {"b": "c", "c": "b"},       1.0),
         (False, False, True,  {"b": "d", "c": "e"},       1.0),
         (False, False, False, {},                         0.0)]
    )
    def test_create_scenarios(self, historical_predictions, xcorr, additional_regressors, expected_extra_regressors,
                                expected_value):

        try:
            os.remove("test_hist_pred_saves/test_create_scenarios.pkl")
        except FileNotFoundError:
            pass

        param_config = {
            "input_parameters": {
                "datetime_column_name": "date",
                "index_column_name": "date",
            },
            "model_parameters": {
                "test_values": 5,
                "delta_training_percentage": 30,
                "prediction_lags": 10,
                "possible_transformations": "none",
                "models": "mockup",
                "main_accuracy_estimator": "mae",
            },
        }

        if historical_predictions:
            param_config["historical_prediction_parameters"] = {
                "initial_index": "2000-01-15",
                "save_path": "test_hist_pred_saves/test_create_scenarios.pkl"
            }

        if xcorr:
            param_config["xcorr_parameters"] = {
                "xcorr_max_lags": 5,
                "xcorr_extra_regressor_threshold": 0.0,  # Force the predictor to use it
                "xcorr_mode": "pearson",
                "xcorr_mode_target": "pearson"
            }

        if additional_regressors:
            param_config["additional_regressors"] = {
                "b": "test_datasets/test_create_scenarios_extrareg_d.csv",
                "c": "test_datasets/test_create_scenarios_extrareg_e.csv",
            }

        ing_data = DataFrame({"a": pandas.date_range('2000-01-01', periods=30),
                              "b": np.arange(30, 60), "c": np.arange(60, 90)})
        ing_data.set_index("a", inplace=True)
        ing_data = add_freq(ing_data, "D")

        scenarios = create_scenarios(ing_data, param_config)

        assert len(scenarios) == 2
        for scenario in scenarios:
            name = scenario.scenario_data.columns[0]

            if expected_extra_regressors != {}:
                assert scenario.models['mockup'].characteristics['extra_regressors'] == expected_extra_regressors[name]

            if historical_predictions:
                hp = scenario.historical_prediction['mockup']
                assert hp.loc[pandas.to_datetime('2000-01-15', format="%Y-%m-%d"):, name].all() == expected_value
            else:
                assert scenario.historical_prediction is None

        try:
            os.remove("test_hist_pred_saves/test_create_scenarios.pkl")
        except FileNotFoundError:
            pass



