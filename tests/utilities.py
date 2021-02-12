import numpy as np
import pandas
from pandas import DataFrame


def get_fake_df(length: int, name: str = "value") -> DataFrame:
    dates = pandas.date_range('1/1/2000', periods=length)

    np.random.seed(0)
    df = pandas.DataFrame(np.random.randn(length), index=dates, columns=[name])
    return df
