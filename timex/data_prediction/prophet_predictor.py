import json
import pkgutil
import logging
import os

from fbprophet import Prophet
import pandas as pd
from pandas import DataFrame
import numpy as np

from timex.data_prediction.data_prediction import PredictionModel, TestingPerformance, pre_transformation, \
    post_transformation

logging.getLogger('fbprophet').setLevel(logging.WARNING)


class FBProphet(PredictionModel):
    """Facebook's Prophet prediction model."""

    def __init__(self, params: dict, transformation: str = None):
        super().__init__(params, name="FBProphet", transformation=transformation)

        # Stuff needed to make Prophet shut up during training.
        self.suppress_stdout_stderr = suppress_stdout_stderr
        self.fbmodel = Prophet()

    def train(self, input_data: DataFrame, extra_regressors: DataFrame = None):
        """Overrides PredictionModel.train()"""
        self.fbmodel = Prophet()

        if extra_regressors is not None:
            input_data = input_data.join(extra_regressors)
            input_data.reset_index(inplace=True)
            column_indices = [0, 1]
            new_names = ['ds', 'y']
            old_names = input_data.columns[column_indices]
            input_data.rename(columns=dict(zip(old_names, new_names)), inplace=True)
            [self.fbmodel.add_regressor(col) for col in extra_regressors.columns]

        else:
            input_data.reset_index(inplace=True)
            input_data.columns = ['ds', 'y']

        with self.suppress_stdout_stderr():
            self.fbmodel.fit(input_data)

    def predict(self, future_dataframe: DataFrame, extra_regressors: DataFrame = None) -> DataFrame:
        """Overrides PredictionModel.predict()"""
        future = self.fbmodel.make_future_dataframe(periods=self.prediction_lags + self.test_values, freq=self.freq)

        if extra_regressors is not None:
            future.set_index('ds', inplace=True)
            future = future.join(extra_regressors.copy())
            future.reset_index(inplace=True)

        forecast = self.fbmodel.predict(future)

        forecast.loc[:, 'yhat'] = post_transformation(forecast['yhat'], self.transformation)
        forecast.loc[:, 'yhat_lower'] = post_transformation(forecast['yhat_lower'], self.transformation)
        forecast.loc[:, 'yhat_upper'] = post_transformation(forecast['yhat_upper'], self.transformation)

        forecast.set_index('ds', inplace=True)

        return forecast


class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)
