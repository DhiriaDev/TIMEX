import numpy as np
import pandas
from pandas import DataFrame


def get_fake_df(length: int, features: int = 1, name: str = "value") -> DataFrame:
    dates = pandas.date_range('1/1/2000', periods=length)

    np.random.seed(0)
    df = pandas.DataFrame({f"{name}_{i}": np.random.randn(length) for i in range(0, features)},
                          index=dates)
    return df

