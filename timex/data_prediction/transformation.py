from pandas import Series
import numpy as np
from scipy.stats import yeojohnson


class Transformation:
    """
    Class used to represent various types of data transformation.
    """

    def apply(self, data: Series) -> Series:
        """
        Apply the transformation on a Pandas Series. Returns the transformed Series.
        Parameters
        ----------
        data : Series
            Data to transform.

        Returns
        -------
        Series
            Transformed data.
        """
        pass

    def inverse(self, data: Series) -> Series:
        """
        Apply the inverse of the transformation on a Pandas Series of transformed values.
        Returns the data re-transformed to the real world.

        Parameters
        ----------
        data : Series
            Data to transform.

        Returns
        -------
        Series
            Transformed data.
        """
        pass


class Log(Transformation):
    def apply(self, data: Series) -> Series:
        return data.apply(lambda x: np.sign(x) * np.log(abs(x)) if abs(x) > 1 else 0)

    def inverse(self, data: Series) -> Series:
        return data.apply(lambda x: np.sign(x) * np.exp(abs(x)))

    def __str__(self):
        return "Log"


class LogModified(Transformation):
    def apply(self, data: Series) -> Series:
        return data.apply(lambda x: np.sign(x) * np.log(abs(x) + 1))

    def inverse(self, data: Series) -> Series:
        return data.apply(lambda x: np.sign(x) * np.exp(abs(x)) - np.sign(x))

    def __str__(self):
        return "modified Log"


class Identity(Transformation):
    def apply(self, data: Series) -> Series:
        return data

    def inverse(self, data: Series) -> Series:
        return data

    def __str__(self):
        return "none"


class YeoJohnson(Transformation):
    def __init__(self):
        self.lmbda = 0

    def apply(self, data: Series) -> Series:
        res, lmbda = yeojohnson(data)
        self.lmbda = lmbda
        return res

    def inverse(self, data: Series) -> Series:
        lmbda = self.lmbda
        x_inv = np.zeros_like(data)
        pos = data >= 0

        # when x >= 0
        if abs(lmbda) < np.spacing(1.):
            x_inv[pos] = np.exp(data[pos]) - 1
        else:  # lmbda != 0
            x_inv[pos] = np.power(data[pos] * lmbda + 1, 1 / lmbda) - 1

        # when x < 0
        if abs(lmbda - 2) > np.spacing(1.):
            x_inv[~pos] = 1 - np.power(-(2 - lmbda) * data[~pos] + 1,
                                       1 / (2 - lmbda))
        else:  # lmbda == 2
            x_inv[~pos] = 1 - np.exp(-data[~pos])

        return Series(x_inv)

    def __str__(self):
        return f"Yeo-Johnson (lambda: {round(self.lmbda, 3)})"


class Diff(Transformation):
    def __init__(self):
        self.first_value = 0

    def apply(self, data: Series) -> Series:
        self.first_value = data[0]
        return data.diff()[1:]

    def inverse(self, data: Series) -> Series:
        return Series(np.r_[self.first_value, data].cumsum())

    def __str__(self):
        return "differentiate (1)"


def transformation_factory(tr_class: str) -> Transformation:
    """
    Given the type of the transformation, encoded as string, return the Transformation object.

    Parameters
    ----------
    tr_class : str
        Transformation type.

    Returns
    -------
    Transformation
        Transformation object.
    """
    if tr_class == "log":
        return Log()
    elif tr_class == "log_modified":
        return LogModified()
    elif tr_class == "none":
        return Identity()
    elif tr_class == "diff":
        return Diff()
    elif tr_class == "yeo_johnson":
        return YeoJohnson()
