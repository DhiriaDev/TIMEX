import unittest

from pandas._libs.tslibs.timestamps import Timestamp

from src.data_ingestion.data_ingestion import data_ingestion
from src.data_preparation.data_preparation import data_selection


class MyTestCase(unittest.TestCase):

    def test_data_preparation_univariate_1(self):

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


    def test_data_preparation_univariate_2(self):
        param_config = {
            "verbose": "yes",
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

    def test_data_preparation_univariate_3(self):
        param_config = {
            "verbose": "yes",
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


if __name__ == '__main__':
    unittest.main()