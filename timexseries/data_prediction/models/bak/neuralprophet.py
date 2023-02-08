# import logging
# import os
# import warnings
#
# from neuralprophet import NeuralProphet
# import pandas as pd
# from pandas import DataFrame
#
# from timexseries.data_prediction import PredictionModel
# logging.getLogger('nprophet').setLevel(logging.WARNING)
# log = logging.getLogger(__name__)
#
# NeuralProphet.set_log_level("ERROR")
#
#
# class NeuralProphetModel(PredictionModel):
#     """Facebook's NeuralProphet prediction model."""
#
#     def __init__(self, params: dict, transformation: str = "none"):
#         super().__init__(params, name="NeuralProphet", transformation=transformation)
#
#         # Stuff needed to make Prophet shut up during training.
#         self.suppress_stdout_stderr = suppress_stdout_stderr
#         try:
#             self.neuralprophet_parameters = params["model_parameters"]["neuralProphet_parameters"]
#         except KeyError:
#             self.neuralprophet_parameters = None
#
#     def train(self, input_data: DataFrame, extra_regressors: DataFrame = None):
#         """Overrides PredictionModel.train()"""
#
#         self.neuralprophetmodel = NeuralProphet()
#         # if self.neuralprophet_parameters is not None:
#         #     try:
#         #         timeseries_name = input_data.columns[0]
#         #         date_format = self.neuralprophet_parameters["holidays_dataframes"]["date_format"]
#         #         holidays = pd.read_csv(self.neuralprophet_parameters["holidays_dataframes"][timeseries_name])
#         #         holidays.loc[:, "ds"].apply(lambda x: pd.to_datetime(x, format=date_format))
#         #         self.neuralprophetmodel = NeuralProphet(holidays=holidays)
#         #         log.debug(f"Using a dataframe for holidays...")
#         #     except KeyError:
#         #         self.neuralprophetmodel = NeuralProphet()
#         #
#         #     try:
#         #         holiday_country = self.neuralprophet_parameters["holiday_country"]
#         #         self.neuralprophetmodel.add_country_holidays(country_name=holiday_country)
#         #         log.debug(f"Set {holiday_country} as country for holiday calendar...")
#         #     except KeyError:
#         #         pass
#         #
#         # else:
#         # self.neuralprophetmodel = NeuralProphet()
#
#         if extra_regressors is not None:
#             # We could apply self.transformation also on the extra regressors.
#             # From tests, it looks like it doesn't change much/it worsens the forecasts.
#             input_data = input_data.join(extra_regressors)
#             input_data.reset_index(inplace=True)
#             column_indices = [0, 1]
#             new_names = ['ds', 'y']
#             old_names = input_data.columns[column_indices]
#             input_data.rename(columns=dict(zip(old_names, new_names)), inplace=True)
#             [self.neuralprophetmodel.add_future_regressor(col) for col in extra_regressors.columns]
#
#         else:
#             input_data.reset_index(inplace=True)
#             input_data.columns = ['ds', 'y']
#
#         with self.suppress_stdout_stderr():
#             self.neuralprophetmodel.fit(input_data, self.freq)
#
#         self.y = input_data
#
#     def predict(self, future_dataframe: DataFrame, extra_regressors: DataFrame = None) -> DataFrame:
#         """Overrides PredictionModel.predict()"""
#         requested_prediction = len(future_dataframe) - len(self.y)
#
#         if extra_regressors is not None:
#             extra_regressors = extra_regressors.iloc[-requested_prediction:, :]
#             extra_regressors = extra_regressors.reset_index()
#             extra_regressors = extra_regressors.drop(columns=["index"])
#
#         future = self.neuralprophetmodel.make_future_dataframe(self.y, periods=requested_prediction,
#                                                                n_historic_predictions=len(self.y),
#                                                                regressors_df=extra_regressors)
#
#         with self.suppress_stdout_stderr():
#             forecast = self.neuralprophetmodel.predict(future)
#
#         forecast = forecast.rename(columns={"yhat1": "yhat"})
#         forecast.set_index('ds', inplace=True)
#
#         return forecast
#
#
# class suppress_stdout_stderr(object):
#     """
#     A context manager for doing a "deep suppression" of stdout and stderr in
#     Python, i.e. will suppress all print, even if the print originates in a
#     compiled C/Fortran sub-function.
#        This will not suppress raised exceptions, since exceptions are printed
#     to stderr just before a script exits, and after the context manager has
#     exited (at least, I think that is why it lets exceptions through).
#
#     """
#
#     def __init__(self):
#         # Open a pair of null files
#         self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
#         # Save the actual stdout (1) and stderr (2) file descriptors.
#         self.save_fds = [os.dup(1), os.dup(2)]
#
#     def __enter__(self):
#         # Assign the null pointers to stdout and stderr.
#         os.dup2(self.null_fds[0], 1)
#         os.dup2(self.null_fds[1], 2)
#
#     def __exit__(self, *_):
#         # Re-assign the real stdout/stderr back to (1) and (2)
#         os.dup2(self.save_fds[0], 1)
#         os.dup2(self.save_fds[1], 2)
#         # Close the null files
#         for fd in self.null_fds + self.save_fds:
#             os.close(fd)
