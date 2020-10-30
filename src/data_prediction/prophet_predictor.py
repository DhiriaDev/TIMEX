import json
import pkgutil
import logging
import os

from fbprophet import Prophet
import pandas as pd
from pandas import DataFrame
import numpy as np

from src.data_prediction.data_prediction import get_training_stats

logging.getLogger('fbprophet').setLevel(logging.WARNING)


class FBProphet():
    """Facebook's Prophet prediction model."""

    def __init__(self, params: dict):
        self.test_values = 0
        self.freq = None

        # Stuff needed to make Prophet shut up during training.
        self.suppress_stdout_stderr = suppress_stdout_stderr

        self.pre_transformation = None
        self.post_transformation = None

        if params["verbose"] == "yes":
            print('-----------------------------------------------------------')
            print('Model_training: creating Prophet model...')

        if "model_parameters" not in params:
            if params["verbose"] == "yes":
                print("Model_training: loading default settings...")
            parsed = pkgutil.get_data(__name__, "default_prediction_parameters/prophet.json")
            model_parameters = json.loads(parsed)
        else:
            if params["verbose"] == "yes":
                print("Model_training: loading user settings...")
            model_parameters = params["model_parameters"]

        self.model = Prophet()
        self.test_percentage = model_parameters["test_percentage"]
        self.prediction_lags = model_parameters["prediction_lags"]

        if model_parameters["transformation"] == "log":
            self.pre_transformation = np.log
            self.post_transformation = np.exp

    def train(self, df: DataFrame) -> dict:
        """Overrides PredictionModel.train()"""
        test_values = int(round(len(df) * (self.test_percentage / 100)))
        self.test_values = test_values
        self.freq = pd.infer_freq(df.index)
        train_ts = df.iloc[:-test_values]
        test_ts = df.iloc[-test_values:]

        train_ts.reset_index(inplace=True)
        train_ts.columns = ['ds', 'y']
        test_ts.reset_index(inplace=True)
        test_ts.columns = ['ds', 'y']

        if self.pre_transformation is not None:
            with pd.option_context('mode.chained_assignment', None):
                tr = self.pre_transformation
                train_ts['y'] = tr(train_ts['y'])

        with self.suppress_stdout_stderr():
            self.model.fit(train_ts)

        forecast = self.predict()
        forecast = forecast.iloc[-self.prediction_lags-test_values:-self.prediction_lags]

        return get_training_stats(test_ts["y"], forecast["yhat"])

    def predict(self) -> DataFrame:
        """Overrides PredictionModel.predict()"""
        future = self.model.make_future_dataframe(periods=self.prediction_lags+self.test_values, freq=self.freq)
        forecast = self.model.predict(future)

        if self.post_transformation is not None:
            tr = self.post_transformation
            forecast['yhat'] = tr(forecast['yhat'])
            forecast['yhat_lower'] = tr(forecast['yhat_lower'])
            forecast['yhat_upper'] = tr(forecast['yhat_upper'])

        forecast.set_index('ds', inplace=True)

        return forecast

    def get_training_parameters(self) -> dict:
        """Overrides PredictionModel.get_training_parameters()"""
        return {
            'test_percentage': self.test_percentage,
            'prediction_lags': self.prediction_lags,
            'transformation': "log"
        }


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
