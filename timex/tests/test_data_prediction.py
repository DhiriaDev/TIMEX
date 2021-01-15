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