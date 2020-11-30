import unittest
from datetime import datetime

from pandas import DataFrame
import numpy as np

from timex.data_prediction.data_prediction import SingleResult, TestingPerformance, ModelResult
from timex.scenario.scenario import Scenario
from timex.utils.utils import prepare_extra_regressor


class MyTestCase(unittest.TestCase):

    def test_prepare_extra_regressors(self):
        ing_data = DataFrame({"a": np.arange(0, 10), "b": np.arange(10, 20)})
        ing_data.set_index("a", inplace=True)

        forecast = DataFrame({"a": np.arange(8, 15), "yhat": np.arange(40, 47)})
        forecast.set_index("a", inplace=True)

        tp = TestingPerformance(first_used_index=0)
        tp.MAE = 0

        model_results = [SingleResult(forecast, tp)]
        model = [ModelResult(model_results, None)]
        scenario = Scenario(ing_data, model, ing_data, None)

        result = prepare_extra_regressor(scenario)

        expected = DataFrame({"a": np.arange(0, 15),
                              "b": np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 42.0, 43.0,
                                             44.0, 45.0, 46.0])})
        expected.set_index("a", inplace=True)

        self.assertTrue(expected.equals(result))


if __name__ == '__main__':
    unittest.main()