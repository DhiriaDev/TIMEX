import unittest
import numpy as np
from pandas import DataFrame

from timex.data_visualization.data_visualization import calc_xcorr

xcorr_modes = ['pearson', 'kendall', 'spearman', 'matlab_normalized']

class MyTestCase(unittest.TestCase):

    def test_calc_xcorr_1(self):
        # Example from https://www.mathworks.com/help/matlab/ref/xcorr.html

        x = [np.power(0.84, n) for n in np.arange(0, 16)]
        y = np.roll(x, 5)

        df = DataFrame(data={"x": x, "y": y})

        xcorr = calc_xcorr("x", df, max_lags=10, modes=xcorr_modes)
        for mode in xcorr:
            self.assertEqual(xcorr[mode].idxmax()[0], -5)

    def test_calc_xcorr_2(self):
        # Shift a sin. Verify that highest correlation is in the correct region.

        # Restrict the delay to less than one period of the sin.
        max_delay = 50 - 1
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x)

        for i in range(-max_delay, max_delay):
            y_delayed = np.roll(y, i)

            df = DataFrame(data={"y": y, "y_delayed": y_delayed})

            xcorr = calc_xcorr("y", df, max_lags=max_delay, modes=xcorr_modes)
            expected_max_lag = -i
            for mode in xcorr:
                self.assertLess(abs(xcorr[mode].idxmax()[0] - expected_max_lag), 4)



