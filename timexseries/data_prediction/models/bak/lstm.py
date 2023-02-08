import logging

from pandas import DataFrame
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from timexseries.data_prediction import PredictionModel

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
    def __init__(self, input_size=1, hidden_layer_size=10):
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


class LSTMModel(PredictionModel):
    """LSTM prediction model."""
    def __init__(self, params: dict, transformation: str = "none"):
        super().__init__(params, name="LSTM", transformation=transformation)

    def predict(self, train_data: DataFrame, points_to_predict: int,
                future_dataframe: DataFrame, extra_regressors: DataFrame = None) -> DataFrame:
        """Overrides PredictionModel.predict()"""
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        scalers = {}

        if extra_regressors is not None:
            # We could apply self.transformation also on the extra regressors.
            # From tests, it looks like it doesn't change much/it worsens the forecasts.
            train_data = train_data.join(extra_regressors)
            n_features = 1 + len(extra_regressors.columns)
            for col in extra_regressors.columns:
                scalers[col] = MinMaxScaler(feature_range=(-1, 1))
                extra_regressors[col] = scalers[col].fit_transform(extra_regressors[[col]])
        else:
            n_features = 1

        train_data.reset_index(inplace=True)
        column_indices = [0, 1]
        new_names = ['ds', 'y']
        old_names = train_data.columns[column_indices]
        train_data.rename(columns=dict(zip(old_names, new_names)), inplace=True)
        train_data.set_index('ds', inplace=True)

        scalers['y'] = MinMaxScaler(feature_range=(-1, 1))
        train_data['y'] = scalers['y'].fit_transform(train_data[['y']])

        n_steps_in, n_steps_out = round(len(train_data)/4), 1

        train_inout_seq = split_sequences(train_data, n_steps_in, n_steps_out, n_features=n_features)

        for i in range(0, len(train_inout_seq)):
            x = np.array(train_inout_seq[i][0], dtype=np.float32)
            y = np.array(train_inout_seq[i][1], dtype=np.float32)
            train_inout_seq[i] = (torch.from_numpy(x), torch.tensor(y))

        model = LSTM(input_size=n_features)
        model.to(dev)

        loss_function = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        epochs = 10

        for i in range(epochs):
            for seq, labels in train_inout_seq:
                seq = seq.to(dev)
                labels = labels.to(dev)
                optimizer.zero_grad()
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(dev),
                                          torch.zeros(1, 1, model.hidden_layer_size).to(dev))

                y_pred = model(seq)

                single_loss = loss_function(y_pred, labels)
                single_loss.backward()
                optimizer.step()

        values_for_prediction = train_inout_seq[-1][0]
        # self.len_train_set = len(train_data)

        if extra_regressors is not None:
            extra_regressors = extra_regressors.iloc[-points_to_predict:].copy()

            for col in extra_regressors:
                scalers[col] = MinMaxScaler(feature_range=(-1, 1))
                extra_regressors[col] = scalers[col].fit_transform(extra_regressors[[col]])

            tensors_to_append = []
            for i in range(0, points_to_predict):
                val = np.array(extra_regressors.iloc[i, :], dtype=np.float32)
                tensors_to_append.append(torch.tensor(val))

        x_input = values_for_prediction
        x_input = x_input.to(dev)
        model.eval()

        for i in range(points_to_predict):
            seq = x_input[i:]
            with torch.no_grad():
                model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                                     torch.zeros(1, 1, model.hidden_layer_size))
                result = model(seq)
                if extra_regressors is not None:
                    result = torch.cat((result, tensors_to_append[i].to(dev)))
                x_input = torch.cat((x_input, result.view(1, -1)))

        results = x_input[-points_to_predict:]
        results = results.to("cpu")
        results = [x[0] for x in results]
        actual_predictions = scalers['y'].inverse_transform(np.array(results).reshape(-1, 1))
        future_dataframe.iloc[-points_to_predict:, 0] = np.array(actual_predictions).flatten()

        return future_dataframe


