import numpy
from pandas import DataFrame
from pandas._libs.tslibs.timestamps import Timestamp

from timex.data_ingestion.data_ingestion import data_ingestion
from timex.data_preparation.data_preparation import data_selection, add_diff_column
from timex.tests.utilities import get_fake_df


class TestDataSelection:
    def test_data_selection_univariate_1(self):
        # Select rows using init datetime.
        param_config = {
            "input_parameters": {
                "datetime_format": "%Y-%m-%d"
            },
            "selection_parameters": {
                "init_datetime": "2000-01-02",
                "end_datetime": "2000-12-09"
            },
        }

        df = get_fake_df(length=3)
        selected_df = data_selection(df, param_config)

        assert selected_df.index.values[0] == Timestamp("2000-01-02")
        assert selected_df.index.values[1] == Timestamp("2000-01-03")

        assert df.iloc[1]["value"] == selected_df.iloc[0]["value"]
        assert df.iloc[2]["value"] == selected_df.iloc[1]["value"]
        assert len(selected_df) == 2

    def test_data_selection_univariate_2(self):
        # Select rows using end datetime.
        param_config = {
            "input_parameters": {
                "datetime_format": "%Y-%m-%d"
            },
            "selection_parameters": {
                "init_datetime": "1999-01-02",
                "end_datetime": "2000-01-02"
            },
        }

        df = get_fake_df(length=3)
        selected_df = data_selection(df, param_config)

        assert selected_df.index.values[0] == Timestamp("2000-01-01")
        assert selected_df.index.values[1] == Timestamp("2000-01-02")

        assert df.iloc[0]["value"] == selected_df.iloc[0]["value"]
        assert df.iloc[1]["value"] == selected_df.iloc[1]["value"]
        assert len(selected_df) == 2

    def test_data_selection_univariate_3(self):
        # Select rows using both init and end time.
        param_config = {
            "input_parameters": {
                "datetime_format": "%Y-%m-%d"
            },
            "selection_parameters": {
                "init_datetime": "2000-01-02",
                "end_datetime": "2000-01-04"
            },
        }

        df = get_fake_df(length=5)
        selected_df = data_selection(df, param_config)

        assert selected_df.index.values[0] == Timestamp("2000-01-02")
        assert selected_df.index.values[1] == Timestamp("2000-01-03")
        assert selected_df.index.values[2] == Timestamp("2000-01-04")

        assert df.iloc[1]["value"] == selected_df.iloc[0]["value"]
        assert df.iloc[2]["value"] == selected_df.iloc[1]["value"]
        assert df.iloc[3]["value"] == selected_df.iloc[2]["value"]

        assert len(selected_df) == 3

    def test_data_selection_univariate_4(self):
        # Select rows based on value.
        param_config = {
            "input_parameters": {
                "source_data_url": "test_datasets/test_1.csv",
                "columns_to_load_from_url": "first_column,third_column",
                "datetime_column_name": "first_column",
                "index_column_name": "first_column",
                "datetime_format": "%Y-%m-%dT%H:%M:%S",
                "frequency": "D"
            },
            "selection_parameters": {
                "column_name_selection": "third_column",
                "value_selection": 3,
            },
        }

        df = data_ingestion(param_config)
        df = data_selection(df, param_config)

        assert df.index.values[0] == Timestamp("2020-02-25")

        assert df.iloc[0]["third_column"] == 3
        assert len(df) == 1


class TestAddDiff:
    def test_add_diff_column_1(self):
        # Add a single diff column.
        df = get_fake_df(3)

        new_df = add_diff_column(df, ["value"])
        assert df.iloc[1]["value"] == new_df.iloc[0]["value"]
        assert df.iloc[2]["value"] == new_df.iloc[1]["value"]

        assert new_df.iloc[0]["value_diff"] == df.iloc[1]["value"]-df.iloc[0]["value"]
        assert new_df.iloc[1]["value_diff"] == df.iloc[2]["value"]-df.iloc[1]["value"]

        assert len(new_df) == 2

    def test_add_diff_column_2(self):
        # Add a multiple diff column.

        df = DataFrame({"a": [0, 1, 2], "b": [10, 30, 60], "c": [5, 10, 20]}, dtype=numpy.float)
        df.set_index("a", inplace=True, drop=True)

        new_df = add_diff_column(df, ["b", "c"])

        test_df = DataFrame({"a": [1, 2], "b": [30, 60], "c": [10, 20], "b_diff": [20, 30], "c_diff": [5, 10]},
                            dtype=numpy.float)
        test_df.set_index("a", inplace=True, drop=True)

        assert new_df.equals(test_df)

    def test_add_diff_column_3(self):
        # Add a multiple diff column. Group by.

        df = DataFrame({"a": [0, 0, 0, 1, 1, 1, 2, 2, 2], "b": [10, 20, 30, 10, 20, 30, 10, 20, 30],
                        "c": [1, 1, 2, 3, 5, 8, 13, 21, 34], "d": [1, 2, 3, 5, 8, 13, 21, 34, 55]}, dtype=numpy.float)
        df.set_index(["a", "b"], inplace=True, drop=True)

        new_df = add_diff_column(df, ["c", "d"], group_by="b")

        test_df = DataFrame({"a": [1, 1, 1, 2, 2, 2], "b": [10, 20, 30, 10, 20, 30],
                             "c": [3, 5, 8, 13, 21, 34], "d": [5, 8, 13, 21, 34, 55],
                             "c_diff": [2, 4, 6, 10, 16, 26], "d_diff": [4, 6, 10, 16, 26, 42]}, dtype=numpy.float)
        test_df.set_index(["a", "b"], inplace=True, drop=True)

        assert new_df.equals(test_df)

