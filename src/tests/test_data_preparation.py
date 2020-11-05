import unittest

from pandas._libs.tslibs.timestamps import Timestamp

from src.data_ingestion.data_ingestion import data_ingestion
from src.data_preparation.data_preparation import data_selection, add_diff_column
from src.tests.utilities import get_fake_df


class MyTestCase(unittest.TestCase):

    def test_data_selection_univariate_1(self):
        # Select rows using init datetime.
        param_config = {
            "verbose": "no",
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

        self.assertEqual(selected_df.index.values[0], Timestamp("2000-01-02"))
        self.assertEqual(selected_df.index.values[1], Timestamp("2000-01-03"))

        self.assertEqual(df.iloc[1]["value"], selected_df.iloc[0]["value"])
        self.assertEqual(df.iloc[2]["value"], selected_df.iloc[1]["value"])
        self.assertEqual(len(selected_df), 2)

    def test_data_selection_univariate_2(self):
        # Select rows using end datetime.
        param_config = {
            "verbose": "no",
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

        self.assertEqual(selected_df.index.values[0], Timestamp("2000-01-01"))
        self.assertEqual(selected_df.index.values[1], Timestamp("2000-01-02"))

        self.assertEqual(df.iloc[0]["value"], selected_df.iloc[0]["value"])
        self.assertEqual(df.iloc[1]["value"], selected_df.iloc[1]["value"])
        self.assertEqual(len(selected_df), 2)

    def test_data_selection_univariate_3(self):
        # Select rows using both init and end time.
        param_config = {
            "verbose": "no",
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

        self.assertEqual(selected_df.index.values[0], Timestamp("2000-01-02"))
        self.assertEqual(selected_df.index.values[1], Timestamp("2000-01-03"))
        self.assertEqual(selected_df.index.values[2], Timestamp("2000-01-04"))

        self.assertEqual(df.iloc[1]["value"], selected_df.iloc[0]["value"])
        self.assertEqual(df.iloc[2]["value"], selected_df.iloc[1]["value"])
        self.assertEqual(df.iloc[3]["value"], selected_df.iloc[2]["value"])

        self.assertEqual(len(selected_df), 3)

    def test_data_selection_univariate_4(self):
        # Select rows based on value.
        param_config = {
            "verbose": "yes",
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

        self.assertEqual(df.index.values[0], Timestamp("2020-02-25"))

        self.assertEqual(df.iloc[0]["third_column"], 3)
        self.assertEqual(len(df), 1)

    def test_add_diff_column_1(self):
        # Select rows based on value.
        param_config = {
            "verbose": "no",
            "input_parameters": {
                "source_data_url": "test_datasets/test_2.csv",
                "columns_to_load_from_url": "first_column,third_column",
                "datetime_column_name": "first_column",
                "index_column_name": "first_column",
                "datetime_format": "%Y-%m-%dT%H:%M:%S"
            },
            "selection_parameters": {
                "column_name_selection": "third_column",
                "value_selection": 3,
            },
        }

        df = data_ingestion(param_config)

        df = add_diff_column(df, "third_column", "test", "no")
        self.assertEqual(df.iloc[0]["third_column"], 8)
        self.assertEqual(df.iloc[1]["third_column"], 15)

        self.assertEqual(df.iloc[0]["test"], 5)
        self.assertEqual(df.iloc[1]["test"], 7)

        self.assertEqual(len(df), 2)


if __name__ == '__main__':
    unittest.main()