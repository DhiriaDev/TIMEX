import os
import logging

from pandas import DataFrame
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from timex.data_prediction.data_prediction import PredictionModel
log = logging.getLogger(__name__)


def split_sequences(df, n_in, n_out, n_features):
    in_out_sequence = []

    for i in range(len(df)):
        # find the end of this pattern
        end_ix = i + n_in
        out_end_ix = end_ix + n_out
        # check if we are beyond the sequence
        if out_end_ix > len(df):
            break
        # gather input and output parts of the pattern
        seq_x = []
        for j in range(i, end_ix):
            this_x = []
            for k in range(0, n_features):
                this_x.append(df.iloc[j, k])
                seq_x.append(this_x)

        seq_y = list(df.iloc[end_ix:out_end_ix, 0].values)
        in_out_sequence.append((seq_x, seq_y))
    return in_out_sequence


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=20):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=1)

        self.linear = nn.Linear(hidden_layer_size, 1)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


class LSTM_model(PredictionModel):
    """LSTM prediction model."""
    def __init__(self, params: dict, transformation: str = None):
        super().__init__(params, name="LSTM", transformation=transformation)

    def train(self, input_data: DataFrame, extra_regressors: DataFrame = None):
        """Overrides PredictionModel.train()"""

        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        if extra_regressors is not None:
            # We could apply self.transformation also on the extra regressors.
            # From tests, it looks like it doesn't change much/it worsens the forecasts.
            input_data = input_data.join(extra_regressors)
            n_features = 1 + len(extra_regressors.columns)
        else:
            n_features = 1

        input_data.reset_index(inplace=True)
        column_indices = [0, 1]
        new_names = ['ds', 'y']
        old_names = input_data.columns[column_indices]
        input_data.rename(columns=dict(zip(old_names, new_names)), inplace=True)
        input_data.set_index('ds', inplace=True)

        self.scalers = {}

        for col in input_data.columns:
            self.scalers[col] = MinMaxScaler(feature_range=(-1, 1))
            input_data[col] = self.scalers[col].fit_transform(input_data[[col]])


        # raw_seq = input_data.iloc[:, 0].values
        n_steps_in, n_steps_out = round(self.delta_training_values/4), 1
        #
        train_inout_seq = split_sequences(input_data, n_steps_in, n_steps_out, n_features=n_features)

        for i in range(0, len(train_inout_seq)):
            x = np.array(train_inout_seq[i][0], dtype=np.float32)
            y = np.array(train_inout_seq[i][1], dtype=np.float32)
            train_inout_seq[i] = (torch.from_numpy(x), torch.tensor(y))

        # train_data_normalized = self.scaler.fit_transform(np.array(raw_seq).reshape(-1, 1))
        # train_data_torch = torch.FloatTensor(train_data_normalized).view(-1)
        #

        # train_inout_seq = split_sequence(train_data_torch, n_steps_in, n_steps_out)

        self.model = LSTM(input_size=n_features)
        self.model.to(dev)

        loss_function = nn.L1Loss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        epochs = 10

        for i in range(epochs):
            for seq, labels in train_inout_seq:
                seq = seq.to(dev)
                labels = labels.to(dev)
                optimizer.zero_grad()
                self.model.hidden_cell = (torch.zeros(1, 1, self.model.hidden_layer_size).to(dev),
                                          torch.zeros(1, 1, self.model.hidden_layer_size).to(dev))

                y_pred = self.model(seq)

                single_loss = loss_function(y_pred, labels)
                single_loss.backward()
                optimizer.step()

        self.values_for_prediction = train_inout_seq[-1][0]
        self.len_train_set = len(input_data)

        # print(f"Len tr set: {self.len_train_set}")
        # print(f"Values for prediction: {self.values_for_prediction}")

        # if extra_regressors is not None:
        #     # We could apply self.transformation also on the extra regressors.
        #     # From tests, it looks like it doesn't change much/it worsens the forecasts.
        #     input_data = input_data.join(extra_regressors)
        #     input_data.reset_index(inplace=True)
        #     column_indices = [0, 1]
        #     new_names = ['ds', 'y']
        #     old_names = input_data.columns[column_indices]
        #     input_data.rename(columns=dict(zip(old_names, new_names)), inplace=True)
        #     [self.fbmodel.add_regressor(col) for col in extra_regressors.columns]

        # else:
        #     input_data.reset_index(inplace=True)
        #     input_data.columns = ['ds', 'y']

        # with self.suppress_stdout_stderr():
        #     self.fbmodel.fit(input_data)

    def predict(self, future_dataframe: DataFrame, extra_regressors: DataFrame = None) -> DataFrame:
        """Overrides PredictionModel.predict()"""
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        # print(f"Predict.")
        requested_prediction = len(future_dataframe) - self.len_train_set

        if extra_regressors is not None:
            extra_regressors = extra_regressors.iloc[-requested_prediction:].copy()

            for col in extra_regressors:
                self.scalers[col] = MinMaxScaler(feature_range=(-1, 1))
                extra_regressors[col] = self.scalers[col].fit_transform(extra_regressors[[col]])

            tensors_to_append = []
            for i in range(0, requested_prediction):
                val = np.array(extra_regressors.iloc[i, :], dtype=np.float32)
                tensors_to_append.append(torch.tensor(val))
            # tensors_to_append.to(dev)

        # print(f"Requested prediction: {requested_prediction}")

        x_input = self.values_for_prediction
        x_input = x_input.to(dev)
        self.model.eval()

        for i in range(requested_prediction):
            seq = x_input[i:]
            with torch.no_grad():
                self.model.hidden = (torch.zeros(1, 1, self.model.hidden_layer_size),
                                     torch.zeros(1, 1, self.model.hidden_layer_size))
                result = self.model(seq)
                if extra_regressors is not None:
                    result = torch.cat((result, tensors_to_append[i].to(dev)))
                #         print(model(seq))
                x_input = torch.cat((x_input, result.view(1, -1)))

        # with torch.no_grad():
        #     results = self.model(x_input)

        results = x_input[-requested_prediction:]
        results = results.to("cpu")
        results = [x[0] for x in results]
        actual_predictions = self.scalers['y'].inverse_transform(np.array(results).reshape(-1, 1))
        future_dataframe.iloc[-requested_prediction:, 0] = np.array(actual_predictions).flatten()

        future_dataframe.loc[:, 'yhat'] = self.transformation.inverse(future_dataframe['yhat'])
        # forecast.loc[:, 'yhat_lower'] = self.transformation.inverse(forecast['yhat_upper'])
        # forecast.loc[:, 'yhat_upper'] = self.transformation.inverse(forecast['yhat_upper'])

        return future_dataframe


