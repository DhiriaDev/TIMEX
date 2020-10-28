import unittest

from pandas._libs.tslibs.timestamps import Timestamp

from src.data_ingestion.data_ingestion import data_ingestion
from src.data_preparation.data_preparation import data_selection, add_diff_column


class MyTestCase(unittest.TestCase):

    def test_data_selection_univariate_1(self):
        # Select rows using init datetime.
        param_config = {
            "verbose": "no",
            "input_parameters": {
                "source_data_url": "test_datasets/test_1.csv",
                "columns_to_load_from_url": "first_column,third_column",
                "datetime_column_name": "first_column",
                "index_column_name": "first_column",
                "datetime_format": "%Y-%m-%dT%H:%M:%S"
            },
            "selection_parameters": {
                "column_name_selection": "",
                "value_selection": "",
                "init_datetime": "2020-02-26T17:00:00",
                "end_datetime": "2020-12-09T17:00:00"
            },
        }

        df = data_ingestion(param_config)
        df = data_selection(df, param_config)

        self.assertEqual(df.index.values[0], Timestamp("2020-02-26 18:00:00"))
        self.assertEqual(df.index.values[1], Timestamp("2020-02-27 18:00:00"))

        self.assertEqual(df.iloc[0]["third_column"], 6)
        self.assertEqual(df.iloc[1]["third_column"], 9)
        self.assertEqual(len(df), 2)

    def test_data_selection_univariate_2(self):
        # Select rows using end datetime.
        param_config = {
            "verbose": "no",
            "input_parameters": {
                "source_data_url": "test_datasets/test_1.csv",
                "columns_to_load_from_url": "first_column,third_column",
                "datetime_column_name": "first_column",
                "index_column_name": "first_column",
                "datetime_format": "%Y-%m-%dT%H:%M:%S"
            },
            "selection_parameters": {
                "init_datetime": "2020-02-22T17:00:00",
                "end_datetime": "2020-02-26T19:00:00"
            },
        }

        df = data_ingestion(param_config)
        df = data_selection(df, param_config)

        self.assertEqual(df.index.values[0], Timestamp("2020-02-25 18:00:00"))
        self.assertEqual(df.index.values[1], Timestamp("2020-02-26 18:00:00"))

        self.assertEqual(df.iloc[0]["third_column"], 3)
        self.assertEqual(df.iloc[1]["third_column"], 6)
        self.assertEqual(len(df), 2)

    def test_data_selection_univariate_3(self):
        # Select rows using both init and end time.
        param_config = {
            "verbose": "no",
            "input_parameters": {
                "source_data_url": "test_datasets/test_1.csv",
                "columns_to_load_from_url": "first_column,third_column",
                "datetime_column_name": "first_column",
                "index_column_name": "first_column",
                "datetime_format": "%Y-%m-%dT%H:%M:%S"
            },
            "selection_parameters": {
                "init_datetime": "2020-02-25T19:00:00",
                "end_datetime": "2020-02-27T17:00:00"
            },
        }

        df = data_ingestion(param_config)
        df = data_selection(df, param_config)

        self.assertEqual(df.index.values[0], Timestamp("2020-02-26 18:00:00"))

        self.assertEqual(df.iloc[0]["third_column"], 6)
        self.assertEqual(len(df), 1)

    def test_data_selection_univariate_4(self):
        # Select rows based on value.
        param_config = {
            "verbose": "no",
            "input_parameters": {
                "source_data_url": "test_datasets/test_1.csv",
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
        df = data_selection(df, param_config)

        self.assertEqual(df.index.values[0], Timestamp("2020-02-25 18:00:00"))

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
        print(df)
        self.assertEqual(df.iloc[0]["third_column"], 8)
        self.assertEqual(df.iloc[1]["third_column"], 15)

        self.assertEqual(df.iloc[0]["test"], 5)
        self.assertEqual(df.iloc[1]["test"], 7)

        self.assertEqual(len(df), 2)


if __name__ == '__main__':
    unittest.main()