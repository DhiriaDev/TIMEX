# import logging
#
# from pandas import DataFrame
# import numpy as np
# from flaml import AutoML
#
# from timexseries.data_prediction import PredictionModel
# log = logging.getLogger(__name__)
#
#
# class FLAMLModel(PredictionModel):
#     """
#     FLAML
#     """
#
#     def __init__(self, params: dict, transformation: str = "none"):
#         super().__init__(params, name="FLAML", transformation=transformation)
#
#         self.len_train_set = 0
#         self.requested_predictions = 0
#         self.train_data = None
#
#     def train(self, input_data: DataFrame, extra_regressors: DataFrame = None):
#         """Overrides PredictionModel.train()"""
#
#         self.len_train_set = len(input_data)
#         self.train_data = input_data
#
#     def predict(self, future_dataframe: DataFrame, extra_regressors: DataFrame = None) -> DataFrame:
#         """Overrides PredictionModel.predict()"""
#         requested_prediction = len(future_dataframe) - self.len_train_set
#         automl = AutoML()
#         settings = {
#             # "time_budget": 10,  # total running time in seconds
#             "metric": 'mae',  # primary metric for validation: 'mape' is generally used for forecast tasks
#             "task": 'ts_forecast',  # task type
#             # "log_file_name": 'CO2_forecast.log',  # flaml log file
#             "eval_method": "auto",  # validation method can be chosen from ['auto', 'holdout', 'cv']
#         }
#         automl.fit(dataframe=self.train_data.reset_index(),  # training data
#                    label=self.train_data.columns[0],  # label column
#                    period=requested_prediction,  # key word argument 'period' must be included for forecast task)
#                    **settings)
#
#         flaml_y_pred = automl.predict(future_dataframe.reset_index().iloc[:, 0].to_frame())
#         future_dataframe.iloc[-requested_prediction:, 0] = flaml_y_pred
#         return future_dataframe
#
#
