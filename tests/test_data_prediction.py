import logging
import os

import dateparser
import pytest

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
from scipy.stats import yeojohnson

from prophet import Prophet

from timexseries import TimeSeriesContainer
from timexseries.data_prediction.models.arima import ARIMAModel
from timexseries.data_prediction.models.exponential_smoothing import ExponentialSmoothingModel
from timexseries.data_prediction.models.linear import LinearModel
# from timexseries.data_prediction.models.flaml_predictor import FLAMLModel
# from timexseries.data_prediction.models.lstm import LSTMModel
from timexseries.data_prediction.models.mockup import MockUpModel
# from timexseries.data_prediction.models.neuralprophet_predictor import NeuralProphetModel
from timexseries.data_prediction.models.persistence import PersistenceModel
from timexseries.data_prediction.models.random_walk_with_drift import RandomWalkWithDriftModel
from timexseries.data_prediction.models.seasonal_persistence import SeasonalPersistenceModel
from timexseries.data_prediction.models.predictor import ModelResult
from timexseries.data_prediction.models.seasonality_estimator import estimate_seasonality
from timexseries.data_prediction.xcorr import calc_xcorr, calc_all_xcorr

from tests.utilities import get_fake_df
from timexseries.data_ingestion import add_freq, ingest_timeseries

from timexseries.data_prediction.pipeline import prepare_extra_regressor, get_best_univariate_predictions, \
    get_best_multivariate_predictions, compute_historical_predictions, get_best_predictions, \
    create_timeseries_containers, get_result_dict
from timexseries.data_prediction.models.prophet import suppress_stdout_stderr, FBProphetModel
from timexseries.data_prediction.transformation import transformation_factory, Identity

xcorr_modes = ['pearson', 'kendall', 'spearman', 'matlab_normalized']

logger = logging.getLogger()
logger.level = logging.DEBUG


# stream_handler = logging.StreamHandler(sys.stdout)
# logger.addHandler(stream_handler)


class TestTransformations:
    @pytest.mark.parametrize(
        "transformation,input_data,output_data,inverse_result,expected_name",
        [("log",
          [-4, -3, -2, -1, 0, 1, 2, 3, 4],
          [-np.log(4), -np.log(3), -np.log(2), -np.log(1), 0, np.log(1), np.log(2), np.log(3), np.log(4)],
          [-4.0, -3.0, -2.0, 0, 0, 0.0, 2.0, 3.0, 4.0],
          "Log"
          ),
         ("log_modified",
          [-4, -3, -2, -1, 0, 1, 2, 3, 4],
          [-np.log(4 + 1), -np.log(3 + 1), -np.log(2 + 1), -np.log(1 + 1), 0, np.log(1 + 1), np.log(2 + 1),
           np.log(3 + 1), np.log(4 + 1)],
          [-4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0],
          "Log_modified"
          ),
         ("none",
          [-4, -3, -2, -1, 0, 1, 2, 3, 4],
          [-4, -3, -2, -1, 0, 1, 2, 3, 4],
          [-4, -3, -2, -1, 0, 1, 2, 3, 4],
          "None"
          ),
         ("yeo_johnson",
          [-4, -3, -2, -1, 0, 1, 2, 3, 4],
          yeojohnson(np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]))[0],
          [-4, -3, -2, -1, 0, 1, 2, 3, 4],
          "Yeo-Johnson (lambda: 1)"
          )]
    )
    def test_transformation(self, transformation, input_data, output_data, inverse_result, expected_name):
        ind = pd.date_range(start="2000-01-01", periods=9, freq="D")
        input_data = Series(np.array(input_data), index=ind)
        output_data = Series(np.array(output_data), index=ind)
        inverse_result = Series(np.array(inverse_result), index=ind)

        tr = transformation_factory(transformation)
        res = tr.apply(input_data)

        for i in ind:
            assert output_data.loc[i] == res.loc[i]

        res = tr.inverse(res)

        for i in ind:
            assert np.isclose(inverse_result.loc[i], res.loc[i])

        assert expected_name == str(expected_name)

    def test_transformation_diff(self):
        s = Series(np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]))
        tr = transformation_factory("diff")

        res = tr.apply(s)

        assert res[1] == 1.0
        assert res[2] == 1.0
        assert res[3] == 1.0
        assert res[4] == 1.0
        assert res[5] == 1.0
        assert res[6] == 1.0
        assert res[7] == 1.0
        assert res[8] == 1.0
        assert tr.first_value == -4

        res = tr.inverse(res)

        assert np.allclose(res, s)

        assert str(tr) == f"differentiate (1)"


class Test_Xcorr:
    @pytest.mark.parametrize(
        "max_lag",
        [90, 100, 110]  # Test the three cases max_lags > / = / < len(ingested_data)
    )
    def test_calc_xcorr_polynomial(self, max_lag):
        # Example from https://www.mathworks.com/help/matlab/ref/xcorr.html, slightly modified
        # The y series is antecedent the x series; therefore, a correlation should be found between x and y.

        x = [np.power(0.94, n) for n in np.arange(0, 100)]
        noise = np.random.normal(0, 1, 100)
        x = x + noise

        y = np.roll(x, -5)
        z = np.roll(x, -10)

        df = DataFrame(data={"x": x, "y": y, "z": z})
        df = df.iloc[0:-10]

        xcorr = calc_xcorr("x", df, max_lags=max_lag, modes=xcorr_modes)
        for mode in xcorr:
            assert xcorr[mode].idxmax()[0] == 5
            assert xcorr[mode].idxmax()[1] == 10

    def test_calc_xcorr_sin(self):
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
                assert abs(xcorr[mode].idxmax()[0] - expected_max_lag) < 4

    @pytest.mark.xfail
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
                assert xcorr[mode].loc[expected_max_lag][0] == 1.0


class Test_Models_General:
    def test_launch_model_validation_percentage(self):
        # Percentages' sum is not 100%; adapt the windows. Use "validation_percentage".
        param_config = {
            "model_parameters": {
                "validation_percentage": 10,
                "delta_training_percentage": 20,
                "forecast_horizon": 10,
                "transformation": "none",
                "main_accuracy_estimator": "mae"
            },
        }

        for n_threads in range(1, 7):
            param_config["max_threads"] = n_threads
            df = get_fake_df(100)
            predictor = MockUpModel(param_config)
            model_result = predictor.launch_model(df.copy(), max_threads=n_threads)

            expected_train_set_lengths = [20, 40, 60, 80, 90]

            assert predictor.validation_values == 10
            assert predictor.delta_training_values == 20
            assert predictor.main_accuracy_estimator == "mae"

            assert len(model_result.results) == 5

            for r in model_result.results:
                prediction = r.prediction
                testing_performances = r.testing_performances
                first_used_index = testing_performances.first_used_index

                used_training_set = df.loc[first_used_index:]
                used_training_set = used_training_set.iloc[:-10]
                assert len(prediction) == len(used_training_set) + 10 + 10
                expected_train_set_lengths.remove(len(used_training_set))

            assert len(expected_train_set_lengths) == 0

    def test_launch_model_float_percentages(self):
        # Percentages' sum is not 100% and values are float; adapt the windows. Use "validation_percentage".
        param_config = {
            "model_parameters": {
                "validation_percentage": 11.7,
                "delta_training_percentage": 18.9,
                "forecast_horizon": 10,
                "transformation": "none",
                "main_accuracy_estimator": "mae"
            },
        }

        for n_threads in range(1, 7):
            param_config["max_threads"] = n_threads
            df = get_fake_df(101)
            predictor = MockUpModel(param_config)
            model_result = predictor.launch_model(df.copy(), max_threads=n_threads)

            expected_train_set_lengths = [19, 38, 57, 76, 89]

            assert predictor.validation_values == 12
            assert predictor.delta_training_values == 19
            assert predictor.main_accuracy_estimator == "mae"

            assert len(model_result.results) == 5

            for r in model_result.results:
                prediction = r.prediction
                testing_performances = r.testing_performances
                first_used_index = testing_performances.first_used_index

                used_training_set = df.loc[first_used_index:]
                used_training_set = used_training_set.iloc[:-12]
                assert len(prediction) == len(used_training_set) + 10 + 12
                expected_train_set_lengths.remove(len(used_training_set))

            assert len(expected_train_set_lengths) == 0

    def test_launch_model_validation_percentage(self):
        # Percentages' sum is not 100%; adapt the windows. Use "validation_percentage".
        param_config = {
            "model_parameters": {
                "validation_percentage": 5,
                "delta_training_percentage": 20,
                "forecast_horizon": 10,
                "transformation": "none",
                "main_accuracy_estimator": "mae"
            },
        }

        for n_threads in range(1, 7):
            df = get_fake_df(100)
            predictor = MockUpModel(param_config)
            model_result = predictor.launch_model(df.copy(), max_threads=n_threads)

            expected_train_set_lengths = [20, 40, 60, 80, 95]

            assert predictor.validation_values == 5
            assert predictor.delta_training_values == 20
            assert predictor.main_accuracy_estimator == "mae"

            assert len(model_result.results) == 5

            for r in model_result.results:
                prediction = r.prediction
                testing_performances = r.testing_performances
                first_used_index = testing_performances.first_used_index

                used_training_set = df.loc[first_used_index:]
                used_training_set = used_training_set.iloc[:-5]
                assert len(prediction) == len(used_training_set) + 5 + 10
                expected_train_set_lengths.remove(len(used_training_set))

            assert len(expected_train_set_lengths) == 0

    def test_launch_percentages_over_100(self):
        # Percentages' sum is over 100%; adapt the window.
        param_config = {
            "model_parameters": {
                "validation_percentage": 5,
                "delta_training_percentage": 100,
                "forecast_horizon": 10,
                "transformation": "none",
                "main_accuracy_estimator": "mae"
            },
        }

        for n_threads in range(1, 3):
            param_config["max_threads"] = n_threads
            df = get_fake_df(100)
            predictor = MockUpModel(param_config)
            model_result = predictor.launch_model(df.copy(), max_threads=n_threads)

            expected_train_set_lengths = [95]

            assert predictor.validation_values == 5
            assert predictor.delta_training_values == 100
            assert predictor.main_accuracy_estimator == "mae"

            assert len(model_result.results) == 1

            for r in model_result.results:
                prediction = r.prediction
                testing_performances = r.testing_performances
                first_used_index = testing_performances.first_used_index

                used_training_set = df.loc[first_used_index:]
                used_training_set = used_training_set.iloc[:-5]
                assert len(prediction) == len(used_training_set) + 5 + 10
                expected_train_set_lengths.remove(len(used_training_set))

            assert len(expected_train_set_lengths) == 0

    @pytest.mark.parametrize("partial", [True, False])
    def test_launch_model_check_default_parameters(self, partial):
        # Check default parameters.
        param_config = {
            "verbose": "no",
        }
        if partial:
            param_config["model_parameters"] = {}

        for n_threads in range(1, 7):
            param_config["max_threads"] = n_threads
            df = get_fake_df(10)
            predictor = MockUpModel(param_config)
            model_result = predictor.launch_model(df.copy(), max_threads=n_threads)

            expected_train_set_lengths = [2, 4, 6, 8, 9]

            assert predictor.validation_values == 1
            assert predictor.delta_training_values == 2
            assert predictor.main_accuracy_estimator == "mae"
            assert type(predictor.transformation) == Identity

            assert len(model_result.results) == 5

            for r in model_result.results:
                prediction = r.prediction
                testing_performances = r.testing_performances
                first_used_index = testing_performances.first_used_index

                used_training_set = df.loc[first_used_index:]
                used_training_set = used_training_set.iloc[:-1]
                assert len(prediction) == len(used_training_set) + 1 + 1
                expected_train_set_lengths.remove(len(used_training_set))

            assert len(expected_train_set_lengths) == 0

    def test_launch_model_with_extra_regressors(self):
        # Test extra-regressors.
        param_config = {
            "model_parameters": {
                "validation_percentage": 10,
                "delta_training_percentage": 20,
                "forecast_horizon": 10,
                "transformation": "none",
                "main_accuracy_estimator": "mae"
            },
        }

        for n_threads in range(1, 7):
            param_config["max_threads"] = n_threads
            df = get_fake_df(10)
            extra_regressor = get_fake_df(20, name="A")

            predictor = MockUpModel(param_config)
            model_result = predictor.launch_model(df.copy(), extra_regressors=extra_regressor)

            expected_train_set_lengths = [2, 4, 6, 8, 9]

            assert predictor.validation_values == 1
            assert predictor.delta_training_values == 2
            assert predictor.main_accuracy_estimator == "mae"
            assert len(model_result.results) == 5

            for r in model_result.results:
                prediction = r.prediction
                testing_performances = r.testing_performances
                first_used_index = testing_performances.first_used_index

                used_training_set = df.loc[first_used_index:]
                used_training_set = used_training_set.iloc[:-1]
                assert len(prediction) == len(used_training_set) + 1 + 10
                expected_train_set_lengths.remove(len(used_training_set))

            assert len(expected_train_set_lengths) == 0

    @pytest.mark.parametrize("set_min", ["a", "_all"])
    @pytest.mark.parametrize("set_max", ["a", "_all"])
    @pytest.mark.parametrize("round_to_integer", ["a", "_all"])
    @pytest.mark.parametrize("n_threads", [1, 2, 3])
    @pytest.mark.parametrize("mockup_confidence", [True, False])
    def test_launch_model_min_max_values(self, set_min, set_max, round_to_integer, n_threads, mockup_confidence):
        # Test min and max values.
        param_config = {
            "max_threads": n_threads,
            "model_parameters": {
                "validation_percentage": 10,
                "delta_training_percentage": 20,
                "forecast_horizon": 10,
                "possible_transformations": "none",
                "models": "mockup",
                "mockup_confidence": mockup_confidence,
                "main_accuracy_estimator": "mae",
            },
        }

        df = DataFrame(data={"ds": pd.date_range('2000-01-01', periods=30),
                             "a": np.arange(0, 30),
                             "b": np.arange(0, 30)})
        df.set_index("ds", inplace=True)

        forced_predictions = Series(data=np.arange(30.5, 70.5, 1), index=pd.date_range('2000-01-01', periods=40),
                                    name="yhat")
        forced_predictions_lower = Series(data=np.arange(30.0, 70.0, 1), index=pd.date_range('2000-01-01', periods=40),
                                          name="yhat_lower")
        forced_predictions_upper = Series(data=np.arange(31.0, 71.0, 1), index=pd.date_range('2000-01-01', periods=40),
                                          name="yhat_upper")
        param_config["model_parameters"]["mockup_forced_predictions"] = forced_predictions

        param_config["model_parameters"]["min_values"] = {set_min: 63}
        param_config["model_parameters"]["max_values"] = {set_max: 67}
        param_config["model_parameters"]["round_to_integer"] = round_to_integer

        timeseries_containers = get_best_predictions(df, param_config)

        indexes_to_check_min = [0] if set_min == "a" else [0, 1]
        indexes_to_check_max = [0] if set_max == "a" else [0, 1]
        indexes_to_round = [0] if round_to_integer == "a" else [0, 1]

        for i in range(0, 2):
            model_result = timeseries_containers[i].models['mockup']  # Column `a` / `b`
            forecasts_to_test = [r.prediction for r in model_result.results]
            forecasts_to_test.append(model_result.best_prediction)

            for r in forecasts_to_test:
                if mockup_confidence:
                    preds = [r['yhat'], r['yhat_lower'], r['yhat_upper']]
                    expected = [forced_predictions, forced_predictions_lower, forced_predictions_upper]
                else:
                    preds = [r['yhat']]
                    expected = [forced_predictions]

                def round_or_not_x(x):
                    if i in indexes_to_round:
                        return round(x)
                    else:
                        return x

                for p, e in zip(preds, expected):

                    if i in indexes_to_check_min and i not in indexes_to_check_max:
                        pd.testing.assert_series_equal(p, e.loc[p.index[0]:].
                                                       apply(lambda x: 63.0 if x <= 63 else round_or_not_x(x)))
                    elif i not in indexes_to_check_min and i in indexes_to_check_max:
                        pd.testing.assert_series_equal(p, e.loc[p.index[0]:].
                                                       apply(lambda x: 67.0 if x >= 67 else round_or_not_x(x)))
                    elif i in indexes_to_check_min and i in indexes_to_check_max:
                        pd.testing.assert_series_equal(p, e.loc[p.index[0]:].
                                                       apply(
                            lambda x: 67.0 if x >= 67 else (63.0 if x <= 63 else round_or_not_x(x))))


class Test_Models_Specific:
    @pytest.mark.parametrize(
        "model_class,check_multivariate",
        [(FBProphetModel, True),
         (ARIMAModel, False),
         (MockUpModel, True), (ExponentialSmoothingModel, False), (PersistenceModel, False),
         (SeasonalPersistenceModel, False), (LinearModel, False),
         (RandomWalkWithDriftModel, False)]
    )
    def test_models(self, model_class, check_multivariate):
        dates = pd.date_range('1/1/2000', periods=100)
        df = pd.DataFrame(np.random.rand(100), index=dates, columns=["value"])
        future_df = pd.DataFrame(index=pd.date_range(freq="1d",
                                                     start=df.index.values[0],
                                                     periods=110),
                                 columns=["yhat"], dtype=df.iloc[:, 0].dtype)

        np.random.seed(42)
        extra_regressors = pd.DataFrame(data={"a": np.random.random(110), "b": np.random.random(110)},
                                        index=pd.date_range(freq="1d",
                                                            start=df.index.values[0],
                                                            periods=110),
                                        dtype=df.iloc[:, 0].dtype)

        model = model_class({})
        model.freq = "1d"

        result = model.predict(df.copy(), 10, future_df.copy())

        for i in range(100, 110):
            assert not np.isnan(result.iloc[i]['yhat'])

        if check_multivariate:
            model = model_class({})
            model.freq = "1d"

            result_with_extra_regressors = model.predict(df.copy(), 10, future_df.copy(), extra_regressors.copy())

            assert not result.equals(result_with_extra_regressors)


class TestGetPredictions:

    def test_prepare_extra_regressors(self):
        ing_data = DataFrame({"a": np.arange(0.0, 10.0), "b": np.arange(10.0, 20.0)})
        ing_data.set_index("a", inplace=True)

        # Simulate the best forecast with overlapping index with time-series data...
        forecast = DataFrame({"a": np.arange(8.0, 15.0), "yhat": np.arange(40.0, 47.0)})
        forecast.set_index("a", inplace=True)

        models = {'fbprophet': ModelResult(None, None, forecast)}
        timeseries_container = TimeSeriesContainer(ing_data, models, None)

        result = prepare_extra_regressor(timeseries_container, 'fbprophet')

        expected = DataFrame({"a": np.arange(0, 15),
                              "b": np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 42.0, 43.0,
                                             44.0, 45.0, 46.0])})
        expected.set_index("a", inplace=True)

        assert expected.equals(result)

    def test_get_best_univariate_and_multivariate_predictions(self):
        # Check that results are in the correct form.

        ing_data = DataFrame({"a": pd.date_range('1/1/2000', periods=30),
                              "b": np.arange(30, 60), "c": np.arange(60, 90)})
        ing_data.set_index("a", inplace=True)
        ing_data = add_freq(ing_data, "D")

        param_config = {
            "xcorr_parameters": {
                "xcorr_max_lags": 120,
                "xcorr_extra_regressor_threshold": 0.0,  # Force predictor to use extra-regressors
                "xcorr_mode": "pearson",
                "xcorr_mode_target": "pearson"
            },
            "model_parameters": {
                "validation_percentage": 2,
                "delta_training_percentage": 20,
                "forecast_horizon": 10,
                "possible_transformations": "log_modified,none",
                "models": "fbprophet,mockup",
                "main_accuracy_estimator": "mae",
            }
        }

        total_xcorr = calc_all_xcorr(ingested_data=ing_data, param_config=param_config)

        best_transformations, timeseries_containers = get_best_univariate_predictions(ing_data, param_config,
                                                                                      total_xcorr)

        assert len(best_transformations) == 2
        assert best_transformations["fbprophet"]["b"] in ["log_modified", "none"]
        assert best_transformations["fbprophet"]["c"] in ["log_modified", "none"]
        assert best_transformations["mockup"]["b"] in ["log_modified", "none"]
        assert best_transformations["mockup"]["c"] in ["log_modified", "none"]

        # Small trick: fool TIMEX in thinking that none is the best transformation for MockUp model. This way
        # we can check its predictions, which are hardcoded and always 0.0 for univariate and len(extra_regressors) for
        # multivariate... with log_modified values would not be exactly len(extra_regressors).
        best_transformations["mockup"]["b"] = "none"
        best_transformations["mockup"]["c"] = "none"

        assert len(timeseries_containers) == 2
        assert timeseries_containers[0].timeseries_data.columns[0] == "b"
        assert timeseries_containers[1].timeseries_data.columns[0] == "c"

        assert len(timeseries_containers[0].models) == 2
        assert len(timeseries_containers[1].models) == 2

        assert timeseries_containers[0].models['mockup'].best_prediction.iloc[
                   -1, 0] == 0.0  # Check predictions are univariate
        assert timeseries_containers[1].models['mockup'].best_prediction.iloc[-1, 0] == 0.0

        timeseries_containers = get_best_multivariate_predictions(best_transformations=best_transformations,
                                                                  ingested_data=ing_data,
                                                                  timeseries_containers=timeseries_containers,
                                                                  param_config=param_config,
                                                                  total_xcorr=total_xcorr)
        assert len(timeseries_containers) == 2
        assert timeseries_containers[0].timeseries_data.columns[0] == "b"
        assert timeseries_containers[1].timeseries_data.columns[0] == "c"

        assert timeseries_containers[0].models['mockup'].best_prediction.iloc[
                   -1, 0] == 1.0  # Check predictions are multivariate
        assert timeseries_containers[1].models['mockup'].best_prediction.iloc[-1, 0] == 1.0

        assert len(timeseries_containers[0].models) == 2
        assert len(timeseries_containers[1].models) == 2

    def test_compute_predictions(self, tmp_path):
        # Check results are in the correct form and test the function to save historic predictions to file.
        # Delta will be 1, by default.
        ing_data = DataFrame({"a": pd.date_range('2000-01-01', periods=30),
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
                "validation_percentage": 2,
                "delta_training_percentage": 20,
                "forecast_horizon": 10,
                "possible_transformations": "log_modified,none",
                "models": "mockup,fbprophet",
                "main_accuracy_estimator": "mae",
            },
            "historical_prediction_parameters": {
                "initial_index": "2000-01-28",
                "save_path": os.path.join(tmp_path, "test1.pkl")
            }
        }

        timeseries_containers = compute_historical_predictions(ingested_data=ing_data, param_config=param_config)

        assert len(timeseries_containers) == 2
        assert timeseries_containers[0].timeseries_data.columns[0] == "b"
        assert timeseries_containers[1].timeseries_data.columns[0] == "c"

        assert len(timeseries_containers[0].models) == 2
        assert len(timeseries_containers[1].models) == 2

        b_old_hist = timeseries_containers[0].historical_prediction
        c_old_hist = timeseries_containers[1].historical_prediction

        for s in timeseries_containers:
            for model in s.historical_prediction:
                hist_prediction = s.historical_prediction[model]
                assert len(hist_prediction) == 2
                assert hist_prediction['series'].index[0] == pd.to_datetime('2000-01-29', format="%Y-%m-%d")
                assert hist_prediction['series'].index[1] == pd.to_datetime('2000-01-30', format="%Y-%m-%d")

        # Simulate a 1-step ahead in time, so we have collected a new point.
        # Note that past values are changed as well, so we will check that TIMEX does not change the old predictions.
        ing_data = DataFrame({"a": pd.date_range('2000-01-01', periods=31),
                              "b": np.arange(20, 51), "c": np.arange(35, 66)})
        ing_data.set_index("a", inplace=True)
        ing_data = add_freq(ing_data, "D")

        # This time historical predictions will be loaded from file.
        timeseries_containers = compute_historical_predictions(ingested_data=ing_data, param_config=param_config)

        for s in timeseries_containers:
            for model in s.historical_prediction:
                hist_prediction = s.historical_prediction[model]['series']
                assert len(hist_prediction) == 3
                assert hist_prediction.index[0] == pd.to_datetime('2000-01-29', format="%Y-%m-%d")
                assert hist_prediction.index[1] == pd.to_datetime('2000-01-30', format="%Y-%m-%d")
                assert hist_prediction.index[2] == pd.to_datetime('2000-01-31', format="%Y-%m-%d")

        # Check that past predictions have not been touched.
        assert b_old_hist['fbprophet']['series'].iloc[0, 0] == timeseries_containers[0].historical_prediction['fbprophet']['series'].iloc[
            0, 0]
        assert b_old_hist['fbprophet']['series'].iloc[1, 0] == timeseries_containers[0].historical_prediction['fbprophet']['series'].iloc[
            1, 0]
        assert b_old_hist['mockup']['series'].iloc[0, 0] == timeseries_containers[0].historical_prediction['mockup']['series'].iloc[0, 0]
        assert b_old_hist['mockup']['series'].iloc[1, 0] == timeseries_containers[0].historical_prediction['mockup']['series'].iloc[1, 0]

        assert c_old_hist['fbprophet']['series'].iloc[0, 0] == timeseries_containers[1].historical_prediction['fbprophet']['series'].iloc[
            0, 0]
        assert c_old_hist['fbprophet']['series'].iloc[1, 0] == timeseries_containers[1].historical_prediction['fbprophet']['series'].iloc[
            1, 0]
        assert c_old_hist['mockup']['series'].iloc[0, 0] == timeseries_containers[1].historical_prediction['mockup']['series'].iloc[0, 0]
        assert c_old_hist['mockup']['series'].iloc[1, 0] == timeseries_containers[1].historical_prediction['mockup']['series'].iloc[1, 0]

    def test_compute_predictions_2(self, tmp_path):

        ing_data = pd.read_csv("test_datasets/test_covid.csv")
        ing_data["data"] = ing_data["data"].apply(lambda x: dateparser.parse(x))
        ing_data.set_index("data", inplace=True, drop=True)
        ing_data = add_freq(ing_data, "D")

        param_config = {
            "input_parameters": {},
            "model_parameters": {
                "validation_percentage": 5,
                "delta_training_percentage": 30,
                "forecast_horizon": 10,
                "possible_transformations": "none",
                "models": "fbprophet",
                "main_accuracy_estimator": "mae",
            },
            "historical_prediction_parameters": {
                "initial_index": "2020-12-08",
                "save_path": os.path.join(tmp_path, "test2.pkl")
            }
        }

        # You can verify with this code that tr_1 is the best training window.
        # validation_percentage = 5
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
        #                                                  periods=len(tr) + validation_percentage + 10),
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
        timeseries_containers = compute_historical_predictions(ingested_data=ing_data, param_config=param_config)

        timeseries_container = timeseries_containers[1]
        training_results = timeseries_container.models['fbprophet'].results
        training_results.sort(key=lambda x: getattr(x.testing_performances, 'MAE'))

        assert historical_prediction.equals(timeseries_container.models['fbprophet'].best_prediction[['yhat']])

        # Make this test with a log_modified

    def test_compute_predictions_3(self, tmp_path):
        # Test with an historical predictions delta > 1
        # This means that historical predictions are not computed starting from initial index 1-step ahead at time,
        # but they are computed every $delta time points.
        ing_data = DataFrame({"a": pd.date_range('2000-01-01', periods=30),
                              "b": np.arange(30, 60), "c": np.arange(60, 90)})
        ing_data.set_index("a", inplace=True)
        ing_data = add_freq(ing_data, "D")

        param_config = {
            "input_parameters": {},
            "model_parameters": {
                "validation_values": 2,
                "delta_training_percentage": 100,
                "forecast_horizon": 10,
                "possible_transformations": "none",
                "models": "fbprophet,mockup",
                "main_accuracy_estimator": "mae",
            },
            "historical_prediction_parameters": {
                "initial_index": "2000-01-20",
                "save_path": os.path.join(tmp_path, "test3.pkl"),
                "delta": 3
            }
        }

        timeseries_containers = compute_historical_predictions(ingested_data=ing_data, param_config=param_config)

        assert len(timeseries_containers) == 2
        assert timeseries_containers[0].timeseries_data.columns[0] == "b"
        assert timeseries_containers[1].timeseries_data.columns[0] == "c"

        assert len(timeseries_containers[0].models) == 2
        assert len(timeseries_containers[1].models) == 2

        for s in timeseries_containers:
            scen_name = s.timeseries_data.columns[0]
            for model in s.historical_prediction:
                hist_prediction = s.historical_prediction[model]['series']
                assert len(hist_prediction) == 10
                id = 0
                for i in pd.date_range('2000-01-21', periods=10):
                    assert hist_prediction.index[id] == i
                    id += 1

            for endpoint in [*pd.date_range('2000-01-20', periods=4, freq="3d")]:
                tr = ing_data.copy()
                fb_tr = tr.loc[:endpoint]
                fb_tr = fb_tr[[scen_name]]
                fbmodel = Prophet()
                fb_tr.reset_index(inplace=True)
                fb_tr.columns = ['ds', 'y']

                with suppress_stdout_stderr():
                    fbmodel.fit(fb_tr)

                future_df = pd.DataFrame(index=pd.date_range(freq="1d",
                                                             start=endpoint + pd.Timedelta(days=1),
                                                             periods=3),
                                         columns=["yhat"])
                future = future_df.reset_index()
                future.rename(columns={'index': 'ds'}, inplace=True)
                forecast = fbmodel.predict(future)
                forecast.set_index('ds', inplace=True)
                expected_hist_pred = forecast.loc[:, 'yhat']
                expected_hist_pred = expected_hist_pred.astype(object)
                expected_hist_pred.rename(scen_name, inplace=True)
                if endpoint == pd.Timestamp('2000-01-29 00:00:00'):  # Last point, remove last 2 points
                    expected_hist_pred = expected_hist_pred.iloc[0:1]

                computed_hist_pred = s.historical_prediction['fbprophet']['series'].loc[
                                     endpoint + pd.Timedelta(days=1):endpoint + pd.Timedelta(days=3), scen_name]

                assert expected_hist_pred.equals(computed_hist_pred)

    def test_get_best_predictions(self):
        # Test that log_modified transformation is applied and that the results are the expected ones.
        # Ideally this should work the same using other models or transformations; it's just to test that pre/post
        # transformations are correctly applied and that predictions are the ones we would obtain manually.
        # It's nice to use Prophet for this because its predictions are deterministic.

        df = DataFrame(data={"ds": pd.date_range('2000-01-01', periods=30),
                             "b": np.arange(30, 60)})

        local_df = df[["ds", "b"]].copy()
        local_df.rename(columns={"b": "y"}, inplace=True)
        local_df['y'] = local_df['y'].apply(lambda x: np.sign(x) * np.log(abs(x) + 1))

        # Compute "best_prediction"
        model = Prophet()
        with suppress_stdout_stderr():
            model.fit(local_df.copy())

        future = model.make_future_dataframe(periods=5)
        expected_best_prediction = model.predict(future)
        expected_best_prediction.loc[:, 'yhat'] = expected_best_prediction['yhat'].apply(
            lambda x: np.sign(x) * np.exp(abs(x)) - np.sign(x))
        expected_best_prediction.set_index("ds", inplace=True)

        # Compute the prediction we should find in model_results.
        model = Prophet()
        with suppress_stdout_stderr():
            model.fit(local_df.iloc[:-5].copy())

        future = model.make_future_dataframe(periods=10)
        expected_test_prediction = model.predict(future)
        expected_test_prediction.loc[:, 'yhat'] = expected_test_prediction['yhat'].apply(
            lambda x: np.sign(x) * np.exp(abs(x)) - np.sign(x))
        expected_test_prediction.set_index("ds", inplace=True)

        # Use TIMEX
        # yhat_lower and yhat_upper are not deterministic. See https://github.com/facebook/prophet/issues/1695
        param_config = {
            "input_parameters": {},
            "model_parameters": {
                "validation_values": 5,
                "delta_training_percentage": 100,
                "forecast_horizon": 5,
                "possible_transformations": "log_modified",
                "models": "fbprophet",
                "main_accuracy_estimator": "mae",
            },
        }

        ingested_data = df[["ds", "b"]].copy()
        ingested_data.set_index("ds", inplace=True)

        timeseries_containers = get_best_predictions(ingested_data, param_config)
        test_prediction = timeseries_containers[0].models['fbprophet'].results[0].prediction
        best_prediction = timeseries_containers[0].models['fbprophet'].best_prediction

        assert best_prediction[['yhat']].equals(expected_best_prediction[['yhat']])
        assert test_prediction[['yhat']].equals(expected_test_prediction[['yhat']])

    def test_no_additional_regressors_found(self):
        # Check that no multivariate predictions are used if no additional regressors are available.
        ing_data = DataFrame({"a": pd.date_range('2000-01-01', periods=30),
                              "b": np.arange(30, 60), "c": np.random.randint(60, 90, 30)})
        ing_data.set_index("a", inplace=True)
        ing_data = add_freq(ing_data, "D")

        param_config = {
            "xcorr_parameters": {
                "xcorr_max_lags": 5,
                "xcorr_extra_regressor_threshold": 1.01,  # Pearson will be < this threshold.
                "xcorr_mode": "pearson",
                "xcorr_mode_target": "pearson"
            },
            "input_parameters": {},
            "model_parameters": {
                "validation_percentage": 2,
                "delta_training_percentage": 20,
                "forecast_horizon": 10,
                "possible_transformations": "log_modified,none",
                "models": "mockup",
                "main_accuracy_estimator": "mae",
            },
        }

        # MockUp prediction models forecasts "0" if used in univariate mode, "number_of_extra_regressors" in
        # multivariate mode.
        timeseries_containers = create_timeseries_containers(ingested_data=ing_data, param_config=param_config)
        for i in pd.date_range(start="2000-01-31", end="2000-02-09", freq="D"):
            assert timeseries_containers[0].models['mockup'].best_prediction.loc[i, "yhat"] == 0.0
            assert timeseries_containers[1].models['mockup'].best_prediction.loc[i, "yhat"] == 0.0


class TestCreateContainers:
    @pytest.mark.parametrize(
        "historical_predictions, xcorr, additional_regressors, expected_extra_regressors, expected_value",
        [(True, True, True, {"b": "c, d", "c": "b, e"}, 2.0),
         (True, True, False, {"b": "c", "c": "b"}, 1.0),
         (True, False, True, {"b": "d", "c": "e"}, 1.0),
         (True, False, False, {}, 0.0),
         (False, True, True, {"b": "c, d", "c": "b, e"}, 2.0),
         (False, True, False, {"b": "c", "c": "b"}, 1.0),
         (False, False, True, {"b": "d", "c": "e"}, 1.0),
         (False, False, False, {}, 0.0)]
    )
    def test_create_containers(self, historical_predictions, xcorr, additional_regressors, expected_extra_regressors,
                               expected_value, tmp_path):

        param_config = {
            "input_parameters": {
                "datetime_column_name": "date",
                "index_column_name": "date",
            },
            "model_parameters": {
                "validation_values": 5,
                "delta_training_percentage": 30,
                "forecast_horizon": 10,
                "possible_transformations": "none",
                "models": "mockup",
                "main_accuracy_estimator": "mae",
            },
        }

        if historical_predictions:
            param_config["historical_prediction_parameters"] = {
                "initial_index": "2000-01-15",
                "save_path": os.path.join(tmp_path, "test_create_containers.pkl")
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
                "b": "test_datasets/test_create_containers_extrareg_d.csv",
                "c": "test_datasets/test_create_containers_extrareg_e.csv",
            }

        # Having values like 30 -> 60 or 60 -> 90 will make multivariate Mockup model always win on the univariate one
        # because it will return the number of used extra-regressors (the more the lower MAE).
        ing_data = DataFrame({"a": pd.date_range('2000-01-01', periods=30),
                              "b": np.arange(30, 60), "c": np.arange(60, 90)})
        ing_data.set_index("a", inplace=True)
        ing_data = add_freq(ing_data, "D")

        timeseries_containers = create_timeseries_containers(ing_data, param_config)

        assert len(timeseries_containers) == 2
        for container in timeseries_containers:
            name = container.timeseries_data.columns[0]

            if xcorr:
                assert type(container.xcorr) == dict

            if expected_extra_regressors != {}:
                assert container.models['mockup'].characteristics['extra_regressors'] == expected_extra_regressors[name]

            if historical_predictions:
                hp = container.historical_prediction['mockup']['series']
                assert hp.loc[pd.to_datetime('2000-01-15', format="%Y-%m-%d"):, name].eq(expected_value).all()
            else:
                assert container.historical_prediction is None

    def test_create_containers_2(self):
        # Test "_all" key for additional regressors.
        param_config = {
            "input_parameters": {
                "datetime_column_name": "date",
                "index_column_name": "date",
            },
            "model_parameters": {
                "validation_values": 5,
                "delta_training_percentage": 30,
                "forecast_horizon": 10,
                "possible_transformations": "none",
                "models": "mockup",
                "main_accuracy_estimator": "mae",
            },
            "additional_regressors": {
                "_all": "test_datasets/test_create_containers_extrareg_d.csv",
            }
        }

        ing_data = DataFrame({"a": pd.date_range('2000-01-01', periods=30),
                              "b": np.arange(30, 60), "c": np.arange(60, 90)})
        ing_data.set_index("a", inplace=True)
        ing_data = add_freq(ing_data, "D")

        timeseries_containers = create_timeseries_containers(ing_data, param_config)
        assert len(timeseries_containers) == 2
        for container in timeseries_containers:
            assert container.models['mockup'].characteristics['extra_regressors'] == "d"

    @pytest.mark.parametrize(
        "xcorr",
        [True, False]
    )
    def test_create_containers_onlyvisual(self, xcorr):

        param_config = {
            "input_parameters": {
                "datetime_column_name": "date",
                "index_column_name": "date",
            },
        }

        if xcorr:
            param_config["xcorr_parameters"] = {
                "xcorr_max_lags": 5,
                "xcorr_extra_regressor_threshold": 0.5,
                "xcorr_mode": "pearson",
                "xcorr_mode_target": "pearson"
            }

        ing_data = DataFrame({"a": pd.date_range('2000-01-01', periods=30),
                              "b": np.arange(30, 60), "c": np.arange(60, 90)})
        ing_data.set_index("a", inplace=True)
        ing_data = add_freq(ing_data, "D")

        timeseries_containers = create_timeseries_containers(ing_data, param_config)

        assert len(timeseries_containers) == 2
        for container in timeseries_containers:
            name = container.timeseries_data.columns[0]
            assert container.models is None
            assert container.historical_prediction is None
            if xcorr:
                assert container.xcorr is not None
            else:
                assert container.xcorr is None
            assert container.timeseries_data.equals(ing_data[[name]])


class TestEstimateSeasonality:
    def test_white_noise(self):
        # No seasonality.
        df = get_fake_df(length=100, features=1)

        assert estimate_seasonality(df) == 1

    def test_weekly(self):
        # Weekly seasonality for a daily series.
        param_config = {
            "input_parameters": {
                "source_data_url": "test_datasets/covid_example_data_ingestion.csv",
                "columns_to_load_from_url": "data,nuovi_positivi",
                "datetime_column_name": "data",
                "index_column_name": "data",
            }
        }
        df = ingest_timeseries(param_config)[2]
        assert estimate_seasonality(df) == 7


class TestGetResultDict:
    def test_get_result_dict_1(self):
        df = get_fake_df(length=100, features=2)
        param_config = {
            "xcorr_parameters": {
                "xcorr_max_lags": 5,
                "xcorr_extra_regressor_threshold": 0.8,
                "xcorr_mode": "pearson",
                "xcorr_mode_target": "pearson"
            },
            "input_parameters": {},
            "model_parameters": {
                "validation_percentage": 2,
                "delta_training_percentage": 20,
                "forecast_horizon": 10,
                "possible_transformations": "log_modified,none",
                "models": "fbprophet,arima,seasonal_persistence,exponentialsmoothing",
                "main_accuracy_estimator": "mae",
            },
        }
        result = get_result_dict(ingested_data=df, param_config=param_config)

        data_df = pd.read_json(result['data'])
        pred_df = pd.read_json(result['best_pred'])
        model_results = result['models_results']

        assert len(data_df) == 100
        assert len(data_df.columns) == 2

        assert len(pred_df) == 10
        assert len(model_results.keys()) == 2

        assert result['freq'] == 'D'

        assert len(model_results['value_0'].keys()) == 6
        assert len(model_results['value_1'].keys()) == 6

        for ts in ['value_0', 'value_1']:
            for name in ['fbprophet', 'arima', 'seasonal_persistence', 'exponentialsmoothing']:
                m = model_results[ts][name]
                assert m['best_training_window_start'] is not None
                assert m['validation_error'] is not None
                assert len(m['performances_with_different_windows']) == 5
