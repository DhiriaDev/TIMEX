import unittest

from pandas._libs.tslibs.timestamps import Timestamp

from src.data_ingestion.data_ingestion import data_ingestion


class MyTestCase(unittest.TestCase):

    def test_data_ingestion_univariate_1(self):

        param_config = {
            "verbose": "no",
            "input_parameters": {
                "source_data_url": "test_datasets/test_1.csv",
                "columns_to_load_from_url": "first_column,third_column",
                "datetime_column_name": "first_column",
                "index_column_name": "first_column",
                "datetime_format": "%Y-%m-%dT%H:%M:%S"
            }
        }

        df = data_ingestion(param_config)
        self.assertEqual(df.index.name, "first_column")
        self.assertEqual(df.index.values[0], Timestamp("2020-02-25 18:00:00"))
        self.assertEqual(df.index.values[1], Timestamp("2020-02-26 18:00:00"))
        self.assertEqual(df.index.values[2], Timestamp("2020-02-27 18:00:00"))

        self.assertEqual(df.columns[0], "third_column")
        self.assertEqual(df.iloc[0]["third_column"], 3)
        self.assertEqual(df.iloc[1]["third_column"], 6)
        self.assertEqual(df.iloc[2]["third_column"], 9)

        print(df.iloc[[0]]["third_column"])

    def test_data_ingestion_univariate_2(self):

        param_config = {
            "verbose": "no",
            "input_parameters": {
                "source_data_url": "test_datasets/test_1.csv",
                "columns_to_load_from_url": "second_column,third_column",
                "index_column_name": "second_column",
            }
        }

        df = data_ingestion(param_config)

        self.assertEqual(df.index.name, "second_column")

        self.assertEqual(df.index.values[0], 2)
        self.assertEqual(df.index.values[1], 5)
        self.assertEqual(df.index.values[2], 8)

        self.assertEqual(df.columns[0], "third_column")
        self.assertEqual(df.iloc[0]["third_column"], 3)
        self.assertEqual(df.iloc[1]["third_column"], 6)
        self.assertEqual(df.iloc[2]["third_column"], 9)


if __name__ == '__main__':
    unittest.main()