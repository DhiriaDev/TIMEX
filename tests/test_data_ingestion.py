import os
from datetime import datetime

import dateparser
import numpy
from pandas import DataFrame
from pandas._libs.tslibs.timestamps import Timestamp
import pandas as pd
import numpy as np

from .context import timexseries

from timexseries.data_ingestion import ingest_timeseries, add_freq, select_timeseries_portion, add_diff_columns
from .utilities import get_fake_df


class TestDataIngestion:
    def test_ingest_timeseries_univariate_1(self):
        # Local load, with datetime. Infer freq.
        param_config = {
            "input_parameters": {
                "source_data_url": "test_datasets/test_1.csv",
                "columns_to_load_from_url": "first_column,third_column",
                "datetime_column_name": "first_column",
                "index_column_name": "first_column",
                "dateparser_options": {
                    "date_formats": ["%Y-%m-%dT%H:%M:%S"]
                }
            }
        }

        df = ingest_timeseries(param_config)
        assert df.index.name == "first_column"
        assert df.index.values[0] == Timestamp("2020-02-25")
        assert df.index.values[1] == Timestamp("2020-02-26")
        assert df.index.values[2] == Timestamp("2020-02-27")

        assert df.columns[0] == "third_column"
        assert df.iloc[0]["third_column"] == 3
        assert df.iloc[1]["third_column"] == 6
        assert df.iloc[2]["third_column"] == 9

        assert df.index.freq == '1d'

    def test_ingest_timeseries_univariate_2(self):
        # Local load, with datetime. Specify freq.
        param_config = {
            "input_parameters": {
                "source_data_url": "test_datasets/test_1_1.csv",
                "columns_to_load_from_url": "first_column,third_column",
                "datetime_column_name": "first_column",
                "index_column_name": "first_column",
                "dateparser_options": {
                    "date_formats": ["%Y-%m-%dT%H:%M:%S"]
                },
                "frequency": "M"
            }
        }

        df = ingest_timeseries(param_config)
        assert df.index.name == "first_column"
        assert df.index.values[0] == Timestamp("2011-01-31 18:00:00")
        assert df.index.values[1] == Timestamp("2011-02-28 18:00:00")
        assert df.index.values[2] == Timestamp("2011-03-31 18:00:00")

        assert df.columns[0] == "third_column"
        assert df.iloc[0]["third_column"] == 3
        assert df.iloc[1]["third_column"] == 6
        assert df.iloc[2]["third_column"] == 9

        assert df.index.freq == 'M'

    # def test_ingest_timeseries_univariate_3(self):
    #     # Local load, with no datetime column.
    #     param_config = {
    #         "input_parameters": {
    #             "source_data_url": "test_datasets/test_1.csv",
    #             "columns_to_load_from_url": "second_column,third_column",
    #             "index_column_name": "second_column"
    #         }
    #     }
    #
    #     df = ingest_timeseries(param_config)
    #
    #     assert df.index.name == "second_column"
    #
    #     assert df.index.values[0] == 2
    #     assert df.index.values[1] == 5
    #     assert df.index.values[2] == 8
    #
    #     assert df.columns[0] == "third_column"
    #     assert df.iloc[0]["third_column"] == 3
    #     assert df.iloc[1]["third_column"] == 6
    #     assert df.iloc[2]["third_column"] == 9

    # def test_ingest_timeseries_univariate_4(self):
    #     # Local load, with no datetime column. Check columns'order is maintained.
    #     param_config = {
    #         "input_parameters": {
    #             "source_data_url": "test_datasets/test_1.csv",
    #             "columns_to_load_from_url": "third_column,second_column,first_column",
    #             "index_column_name": "second_column",
    #         }
    #     }
    #
    #     df = ingest_timeseries(param_config)
    #
    #     assert df.index.name == "second_column"
    #     assert df.columns[0] == "third_column"
    #     assert df.columns[1] == "first_column"

    def test_ingest_timeseries_univariate_5(self):
        # Local load, with diff columns.
        param_config = {
            "input_parameters": {
                "source_data_url": "test_datasets/test_1_2.csv",
                "columns_to_load_from_url": "third_column,second_column,first_column",
                "datetime_column_name": "first_column",
                "index_column_name": "first_column",
                "dateparser_options": {
                    "date_formats": ["%Y-%m-%dT%H:%M:%S"]
                },
                "add_diff_column": "third_column,second_column"
            }
        }

        df = ingest_timeseries(param_config)

        assert df.index.name == "first_column"
        assert df.columns[0] == "third_column"
        assert df.columns[1] == "second_column"
        assert df.columns[2] == "third_column_diff"
        assert df.columns[3] == "second_column_diff"
        assert len(df.columns) == 4
        assert len(df) == 3

    def test_ingest_timeseries_univariate_6(self):
        # Local load, with diff columns. Rename columns.
        param_config = {
            "input_parameters": {
                "source_data_url": "test_datasets/test_1_2.csv",
                "columns_to_load_from_url": "third_column,second_column,first_column",
                "datetime_column_name": "first_column",
                "index_column_name": "first_column",
                "dateparser_options": {
                    "date_formats": ["%Y-%m-%dT%H:%M:%S"]
                },
                "add_diff_column": "third_column,second_column",
                "timeseries_names":
                    {
                        "first_column": "A",
                        "second_column": "B",
                        "third_column": "C",
                        "third_column_diff": "D",
                        "second_column_diff": "E"
                    }
            }
        }

        df = ingest_timeseries(param_config)

        assert df.index.name == "A"
        assert df.columns[0] == "C"
        assert df.columns[1] == "B"
        assert df.columns[2] == "D"
        assert df.columns[3] == "E"
        assert len(df.columns) == 4
        assert len(df) == 3

    def test_ingest_timeseries_univariate_7(self):
        # Test that duplicated data is removed.
        param_config = {
            "input_parameters": {
                "source_data_url": "test_datasets/test_4.csv",
                "columns_to_load_from_url": "first_column,second_column",
                "datetime_column_name": "first_column",
                "index_column_name": "first_column",
                "dateparser_options": {
                    "date_formats": ["%Y-%m-%dT%H:%M:%S"]
                },
            }
        }

        df = ingest_timeseries(param_config)

        assert len(df) == 3
        assert df.iloc[0, 0] == 1
        assert df.iloc[1, 0] == 3
        assert df.iloc[2, 0] == 4

    def test_ingest_timeseries_univariate_8(self):
        # Local load, read all columns.
        param_config = {
            "input_parameters": {
                "source_data_url": "test_datasets/test_1.csv",
                "columns_to_load_from_url": "",
                "datetime_column_name": "first_column",
                "index_column_name": "first_column",
                "dateparser_options": {
                    "date_formats": ["%Y-%m-%dT%H:%M:%S"]
                }
            }
        }

        df = ingest_timeseries(param_config)
        assert df.index.name == "first_column"
        assert df.index.values[0] == Timestamp("2020-02-25")
        assert df.index.values[1] == Timestamp("2020-02-26")
        assert df.index.values[2] == Timestamp("2020-02-27")

        assert df.columns[0] == "second_column"
        assert df.iloc[0]["second_column"] == 2
        assert df.iloc[1]["second_column"] == 5
        assert df.iloc[2]["second_column"] == 8

        assert df.columns[1] == "third_column"
        assert df.iloc[0]["third_column"] == 3
        assert df.iloc[1]["third_column"] == 6
        assert df.iloc[2]["third_column"] == 9

        assert df.index.freq == '1d'

    def test_ingest_timeseries_univariate_9(self):
        # Local load, with datetime with italian months (e.g. Gen, Feb, etc.)
        # Check that monthly freq is applied.
        param_config = {
            "input_parameters": {
                "source_data_url": "test_datasets/test_5.csv",
                "columns_to_load_from_url": "first_column,third_column",
                "datetime_column_name": "first_column",
                "index_column_name": "first_column",
                "dateparser_options": {
                    "settings": { "PREFER_DAY_OF_MONTH": "first" }
                }
            }
        }

        df = ingest_timeseries(param_config)
        assert df.index.name == "first_column"
        assert df.index.values[0] == Timestamp("2020-11-01")
        assert df.index.values[1] == Timestamp("2020-12-01")
        assert df.index.values[2] == Timestamp("2021-01-01")

        assert df.columns[0] == "third_column"
        assert df.iloc[0]["third_column"] == 3
        assert df.iloc[1]["third_column"] == 6
        assert df.iloc[2]["third_column"] == 9

        assert df.index.freq == 'MS'

    def test_ingest_timeseries_univariate_10(self):
        # Check that data is interpolated.
        param_config = {
            "input_parameters": {
                "source_data_url": "test_datasets/test_6.csv",
                "columns_to_load_from_url": "first_column,second_column",
                "datetime_column_name": "first_column",
                "index_column_name": "first_column",
            }
        }

        df = ingest_timeseries(param_config)
        assert df.index.name == "first_column"
        # assert df.index.values[0] == Timestamp("2020-11-01")
        # assert df.index.values[1] == Timestamp("2020-12-01")
        # assert df.index.values[2] == Timestamp("2021-01-01")
        #
        # assert df.columns[0] == "third_column"
        # assert df.iloc[0]["third_column"] == 3
        # assert df.iloc[1]["third_column"] == 6
        # assert df.iloc[2]["third_column"] == 9

        assert df.index.freq == 'D'
        assert df.iloc[:, 0].isnull().sum() == 0

    def test_ingest_timeseries_univariate_11(self, tmp_path):
        # Check that no dataset exits ingest_timeseries without a frequency and with nan values.
        param_config = {
            "input_parameters": {
                "source_data_url": os.path.join(tmp_path, "test.csv"),
                "columns_to_load_from_url": "date,a,b",
                "datetime_column_name": "date",
                "index_column_name": "date",
            }
        }

        ing_df = pd.read_csv("test_datasets/test_7.csv")
        ing_df['date'] = ing_df['date'].apply(lambda x: dateparser.parse(x))
        ing_df.set_index("date", inplace=True, drop=True)

        for i in range(0, 40):
            df = ing_df.copy()
            for j in range(0, i):
                df = df.drop(df.sample(1).index)

            df.to_csv(os.path.join(tmp_path, "test.csv"))
            df = ingest_timeseries(param_config)

            assert df.iloc[:, 0].isnull().sum() == 0
            assert df.index.freq == "D"

    def test_ingest_timeseries_univariate_12(self):
        # Check that all columns are loaded if `columns_to_load_from_url` is not specified.
        param_config = {
            "input_parameters": {
                "source_data_url": "test_datasets/test_1.csv",
                "datetime_column_name": "first_column",
                "index_column_name": "first_column",
                "dateparser_options": {
                    "date_formats": ["%Y-%m-%dT%H:%M:%S"]
                }
            }
        }

        df = ingest_timeseries(param_config)
        assert len(df.columns) == 2
        assert df.index.name == "first_column"
        assert df.columns[0] == "second_column"
        assert df.columns[1] == "third_column"
        assert len(df) == 3

        assert df.index.freq == '1d'


class TestAddFreq:
    def test_add_freq_1(self):
        # df already has freq; do nothing.
        df = get_fake_df(10)
        new_df = add_freq(df)

        assert df.equals(new_df)

    def test_add_freq_2(self):
        # df is daily; check if it is set, without passing it.
        df = get_fake_df(10)
        df.index.freq = None

        new_df = add_freq(df)
        assert df.equals(new_df)
        assert new_df.index.freq == "D"

    def test_add_freq_3(self):
        # df is daily; check if it is set, passing it.
        df = get_fake_df(10)
        df.index.freq = None

        new_df = add_freq(df, "D")
        assert df.equals(new_df)
        assert new_df.index.freq == "D"

    def test_add_freq_4(self):
        # df is daily, but with different hours.
        # Check if it is set so.
        dates = [pd.Timestamp(datetime(year=2020, month=1, day=1, hour=10, minute=00)),
                 pd.Timestamp(datetime(year=2020, month=1, day=2, hour=12, minute=21)),
                 pd.Timestamp(datetime(year=2020, month=1, day=3, hour=13, minute=30)),
                 pd.Timestamp(datetime(year=2020, month=1, day=4, hour=11, minute=32))]

        ts = pd.DataFrame(np.random.randn(4), index=dates)

        new_ts = add_freq(ts)
        assert ts.iloc[0].equals(new_ts.iloc[0])
        assert new_ts.index[0] == Timestamp('2020-01-01 00:00:00', freq='D')
        assert new_ts.index[1] == Timestamp('2020-01-02 00:00:00', freq='D')
        assert new_ts.index[2] == Timestamp('2020-01-03 00:00:00', freq='D')
        assert new_ts.index[3] == Timestamp('2020-01-04 00:00:00', freq='D')
        assert new_ts.index.freq == "D"

    def test_add_freq_5(self):
        # df has no clear frequency.
        # Check if it is set daily.
        dates = [pd.Timestamp(datetime(year=2020, month=1, day=1, hour=10, minute=00)),
                 pd.Timestamp(datetime(year=2020, month=1, day=3, hour=12, minute=21)),
                 pd.Timestamp(datetime(year=2020, month=1, day=7, hour=13, minute=30)),
                 pd.Timestamp(datetime(year=2020, month=1, day=19, hour=11, minute=32))]

        ts = pd.DataFrame(np.random.randn(4), index=dates)

        new_ts = add_freq(ts)
        assert new_ts.index.freq == "D"


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
        selected_df = select_timeseries_portion(df, param_config)

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
        selected_df = select_timeseries_portion(df, param_config)

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
        selected_df = select_timeseries_portion(df, param_config)

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

        df = ingest_timeseries(param_config)
        df = select_timeseries_portion(df, param_config)

        assert df.index.values[0] == Timestamp("2020-02-25")

        assert df.iloc[0]["third_column"] == 3
        assert len(df) == 1


class TestAddDiff:
    def test_add_diff_column_1(self):
        # Add a single diff column.
        df = get_fake_df(3)

        new_df = add_diff_columns(df, ["value"])
        assert df.iloc[1]["value"] == new_df.iloc[0]["value"]
        assert df.iloc[2]["value"] == new_df.iloc[1]["value"]

        assert new_df.iloc[0]["value_diff"] == df.iloc[1]["value"]-df.iloc[0]["value"]
        assert new_df.iloc[1]["value_diff"] == df.iloc[2]["value"]-df.iloc[1]["value"]

        assert len(new_df) == 2

    def test_add_diff_column_2(self):
        # Add a multiple diff column.

        df = DataFrame({"a": [0, 1, 2], "b": [10, 30, 60], "c": [5, 10, 20]}, dtype=numpy.float)
        df.set_index("a", inplace=True, drop=True)

        new_df = add_diff_columns(df, ["b", "c"])

        test_df = DataFrame({"a": [1, 2], "b": [30, 60], "c": [10, 20], "b_diff": [20, 30], "c_diff": [5, 10]},
                            dtype=numpy.float)
        test_df.set_index("a", inplace=True, drop=True)

        assert new_df.equals(test_df)

    def test_add_diff_column_3(self):
        # Add a multiple diff column. Group by.

        df = DataFrame({"a": [0, 0, 0, 1, 1, 1, 2, 2, 2], "b": [10, 20, 30, 10, 20, 30, 10, 20, 30],
                        "c": [1, 1, 2, 3, 5, 8, 13, 21, 34], "d": [1, 2, 3, 5, 8, 13, 21, 34, 55]}, dtype=numpy.float)
        df.set_index(["a", "b"], inplace=True, drop=True)

        new_df = add_diff_columns(df, ["c", "d"], group_by="b")

        test_df = DataFrame({"a": [1, 1, 1, 2, 2, 2], "b": [10, 20, 30, 10, 20, 30],
                             "c": [3, 5, 8, 13, 21, 34], "d": [5, 8, 13, 21, 34, 55],
                             "c_diff": [2, 4, 6, 10, 16, 26], "d_diff": [4, 6, 10, 16, 26, 42]}, dtype=numpy.float)
        test_df.set_index(["a", "b"], inplace=True, drop=True)

        assert new_df.equals(test_df)