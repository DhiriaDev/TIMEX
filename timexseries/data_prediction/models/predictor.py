import json
import logging
import math
import multiprocessing
import pkgutil
from functools import reduce

import pandas as pd
from pandas import DataFrame

from timexseries.data_prediction.transformation import transformation_factory
from timexseries.data_prediction.validation_performances import ValidationPerformance

log = logging.getLogger(__name__)


class SingleResult:
    """
    Class for the result of a model, trained on a specific training set.

    Parameters
    ----------
    prediction : DataFrame
        Estimated prediction, using this training set
    testing_performances : ValidationPerformance
        Testing performance (`timexseries.data_prediction.validation_performances.ValidationPerformance`), on the validation
        set, obtained using this training set to train the model.
    """

    def __init__(self, prediction: DataFrame, testing_performances: ValidationPerformance):
        self.prediction = prediction
        self.testing_performances = testing_performances


class ModelResult:
    """
    Class for to collect the global results of a model trained on a time-series.

    Parameters
    ----------
    results : [SingleResult]
        List of all the results obtained using all the possible training set for this model, on the time series.
        This is useful to create plots which show how the performance vary changing the training data (e.g.
        `timexseries.data_visualization.functions.performance_plot`).
    characteristics : dict
        Model parameters. This dictionary collects human-readable characteristics of the model, e.g. the used number of
        validation points used, the length of the sliding training window, etc.
    best_prediction : DataFrame
        Prediction obtained using the best training window and _all_ the available points in the time-series. This is
        the prediction that users are most likely to want.
    """

    def __init__(self, results: [SingleResult], characteristics: dict, best_prediction: DataFrame):
        self.results = results
        self.characteristics = characteristics
        self.best_prediction = best_prediction


class PredictionModel:
    """
    Base class for every prediction model which can be used on a time series.

    Parameters
    ----------
    params : dict
        A dictionary corresponding to a TIMEX JSON configuration file.
        The various attributes, described below, will be extracted from this.
        If `params` does not contain the entry `model_parameters`, then TIMEX will attempt to load a default
        configuration parameter dictionary.
    name : str
        Class of the model.
    transformation : str, None, optional
        The class of transformation which the model should use. If not specified, the one in `params` will be used.

    Attributes
    ----------
    freq : str
        If available, the frequency of the time-series.
    test_values : int
        Number of the last points of the time-series to use for the validation set. Default 0. If this is not available
        in the configuration parameter dictionary, `test_percentage` will be used.
    test_percentage : float
        Percentage of the time-series length to used for the validation set. Default 0
    transformation : str
        Transformation to apply to the time series before using it. Default None
    prediction_lags : int
        Number of future lags for which the prediction has to be made. Default 0
    delta_training_percentage : float
        Length, in percentage of the time-series length, of the training windows.
    delta_training_values : int
        Length of the training windows, obtained by computing `delta_training_percentage * length of the time-series`.
    main_accuracy_estimator : str
        Error metric to use when deciding which prediction is better. Default: MAE.
    model_characteristics : dict
        Dictionary of values containing the main characteristics and parameters of the model. Default {}
    """

    def __init__(self, params: dict, name: str, transformation: str = None) -> None:
        self.name = name

        log.info(f"Creating a {self.name} model...")

        if "model_parameters" not in params:
            log.debug(f"Loading default settings...")
            parsed = pkgutil.get_data(__name__, "default_prediction_parameters/" + self.name + ".json")
            model_parameters = json.loads(parsed)
        else:
            log.debug(f"Loading user settings...")
            model_parameters = params["model_parameters"]

        if "test_values" in model_parameters:
            self.test_values = model_parameters["test_values"]
        else:
            self.test_percentage = model_parameters["test_percentage"]
            self.test_values = -1

        if transformation is not None:
            self.transformation = transformation_factory(transformation)
        else:
            self.transformation = transformation_factory(model_parameters["transformation"])

        self.prediction_lags = model_parameters["prediction_lags"]
        self.delta_training_percentage = model_parameters["delta_training_percentage"]
        self.main_accuracy_estimator = model_parameters["main_accuracy_estimator"]
        self.delta_training_values = 0
        self.model_characteristics = {}

        self.freq = ""
        log.debug(f"Finished model creation.")

    def train(self, ingested_data: DataFrame, extra_regressor: DataFrame = None):
        """
        Train the model on the first column of `ingested_data`. Additional time-series, which will be used to improve
        the training in the case of a multivariate model can be passed in the `extra_regressor` DataFrame, one for each
        column.

        The predictor, after launching `train`, is ready to make predictions for the future.

        Note that this and `predict` do not split anything in training data/validation data; this is done with other
        functions, like `launch_model`.

        Parameters
        ----------
        ingested_data : DataFrame
            Training set. The entire time-series in the first column of `ingested_data` will be used for training.

        extra_regressor : DataFrame, optional, default None
            Additional time-series to use for better predictions.

        Examples
        --------
        We will use as example the `timexseries.data_prediction.models.prophet_predictor.FBProphetModel`, an instance of
        `Predictor`.

        >>> param_config = {}  # This will make the predictor use default values...
        >>> predictor = FBProphetModel(params=param_config)

        Create some training data.

        >>> dates = pd.date_range('2000-01-01', periods=30)  # Last index is 2000-01-30
        >>> ds = pd.DatetimeIndex(dates, freq="D")
        >>> a = np.arange(30, 60)
        >>> training_dataframe = DataFrame(data={"a": a}, index=ds)

        Train the model.

        >>> predictor.train(training_dataframe)
        """
        pass

    def predict(self, future_dataframe: DataFrame, extra_regressor: DataFrame = None) -> DataFrame:
        """
        Return a DataFrame with the shape of `future_dataframe`, filled with predicted values.

        This function is used by `compute_training`; `future_dataframe` has the length of the time-series used for
        training, plus the number of desired prediction points.

        Additional `extra_regressor`, which will contain additional time-series useful to improve the prediction in case
        of multivariate models, must have the same points of `future_dataframe`.

        Parameters
        ----------
        future_dataframe : DataFrame
            DataFrame which will be filled with prediction values. This DataFrame should have the same index of the data
            used for training (plus the number of predicted values), and a column named `yhat`, which corresponds to the
            prediction that should be computed.

        extra_regressor : DataFrame, optional, default None
            DataFrame which contain optional time-series which will be used to improve the prediction. It should have
            the same shape of `future_dataframe` but with one column for each time-series. All the values should be
            present.

        Returns
        -------
        forecast : DataFrame
            DataFrame which contains the prediction computed by the model, in the column `yhat`. `forecast` may contain
            additional columns, e.g. `yhat_lower`, `yhat_upper` etc.

        Examples
        --------
        We will use as example the `timexseries.data_prediction.models.prophet_predictor.FBProphetModel`, an instance of
        `Predictor`.
        If the model has been trained, as shown in `predict` example, we can create a forecast.

        First, create the future dataframe which will be filled:

        >>> future_dates = pd.date_range('2000-01-31', periods=7)
        >>> future_ds = pd.DatetimeIndex(future_dates, freq="D")
        >>> future_df = DataFrame(columns=["yhat"], index=future_ds)
        >>> future_df
                   yhat
        ds
        2000-01-31  NaN
        2000-02-01  NaN
        2000-02-02  NaN
        2000-02-03  NaN
        2000-02-04  NaN
        2000-02-05  NaN
        2000-02-06  NaN

        Now we can create the forecast:

        >>> predictions = predictor.predict(future_dataframe=future_df)
        >>> predictions
        ds
        2000-01-31    59.992592
        2000-02-01    60.992592
        2000-02-02    61.992592
        2000-02-03    62.992592
        2000-02-04    63.992592
        2000-02-05    64.992592
        2000-02-06    65.992592
        Name: yhat, dtype: float64
        """
        pass

    def _compute_trainings(self, train_ts: DataFrame, test_ts: DataFrame, extra_regressors: DataFrame, max_threads: int):
        """
        Compute the training of a model on a set of different training sets, of increasing length.
        `train_ts` is split in `n` different training sets, according to the length of `train_ts` and the value of
        `self.delta_training_values`. The computation is split across different processes, according to the value of
        max_threads which indicates the maximum number of usable processes.

        Parameters
        ----------
        train_ts : DataFrame
            The entire training set which can be used; it will be split in different training sets, in order to test
            which sub-training-set performs better.
        test_ts : DataFrame
            Testing set to be used to compute the models' performances.
        extra_regressors : DataFrame
            Additional time-series to pass to `train` in order to improve the performances.
        max_threads : int
            Maximum number of threads to use in the training phase.

        Returns
        -------
        results : list
            List of SingleResult. Each one is the result relative to the use of a specific train set.
        """
        train_sets_number = math.ceil(len(train_ts) / self.delta_training_values)
        log.info(f"Model will use {train_sets_number} different training sets...")

        def c(targets: [int], _return_dict: dict, thread_number: int):
            _results = []

            for _i in range(targets[0], targets[1]):
                tr = train_ts.iloc[-(_i+1) * self.delta_training_values:]

                log.debug(f"Trying with last {len(tr)} values as training set, in thread {thread_number}")

                self.train(tr.copy(), extra_regressors)

                future_df = pd.DataFrame(index=pd.date_range(freq=self.freq,
                                                             start=tr.index.values[0],
                                                             periods=len(tr) + self.test_values + self.prediction_lags),
                                         columns=["yhat"], dtype=tr.iloc[:, 0].dtype)

                forecast = self.predict(future_df, extra_regressors)

                forecast.loc[:, 'yhat'] = self.transformation.inverse(forecast['yhat'])
                try:
                    forecast.loc[:, 'yhat_lower'] = self.transformation.inverse(forecast['yhat_lower'])
                    forecast.loc[:, 'yhat_upper'] = self.transformation.inverse(forecast['yhat_upper'])
                except:
                    pass

                testing_prediction = forecast.iloc[-self.prediction_lags - self.test_values:-self.prediction_lags]

                first_used_index = tr.index.values[0]

                tp = ValidationPerformance(first_used_index)
                tp.set_testing_stats(test_ts.iloc[:, 0], testing_prediction["yhat"])
                _results.append(SingleResult(forecast, tp))

            _return_dict[thread_number] = _results

        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        processes = []

        if self.name == 'LSTM' or self.name == 'NeuralProphet':
            log.info(f"LSTM/NeuralProphet model. Cant use multiprocessing.")
            return_d = {}
            distributions = [[0, train_sets_number]]
            c(distributions[0], return_d, 0)
            return return_d[0]

        if max_threads == 1:
            distributions = [[0, train_sets_number]]
            n_threads = 1
        else:
            distributions = []

            if train_sets_number % max_threads == 0:
                n_threads = max_threads
                subtraining_dim = train_sets_number // n_threads
                for i in range(0, n_threads):
                    distributions.append([i*subtraining_dim, i*subtraining_dim + subtraining_dim])
            else:
                n_threads = min(max_threads, train_sets_number)
                subtraining_dim = train_sets_number // n_threads
                for i in range(0, n_threads):
                    distributions.append([i*subtraining_dim, i*subtraining_dim+subtraining_dim])
                for k in range(0, (train_sets_number % n_threads)):
                    distributions[k][1] += 1
                    distributions[k+1::] = [ [x+1, y+1] for x, y in distributions[k+1::]]

        for i in range(0, n_threads):
            processes.append(multiprocessing.Process(target=c, args=(distributions[i], return_dict, i)))
            processes[-1].start()

        for p in processes:
            p.join()

        results = reduce(lambda x, y: x+y, [return_dict[key] for key in return_dict])

        return results

    def _compute_best_prediction(self, ingested_data: DataFrame, training_results: [SingleResult],
                                 extra_regressors: DataFrame = None):
        """
        Given the ingested data and the training results, identify the best training window and compute a prediction
        using all the possible data, till the end of the series (hence, including the validation set).
        Parameters
        ----------
        ingested_data : DataFrame
            Initial time-series data, in a DataFrame. The first column of the DataFrame is the time-series.
        training_results : [SingleResult]
            List of `SingleResult` object: each one is the result of the model on a specific training-set.
        extra_regressors : DataFrame, optional, default None
            Additional time-series to use for better predictions.
        Returns
        -------
        DataFrame
            Best available prediction for this time-series, with this model.
        """
        training_results.sort(key=lambda x: getattr(x.testing_performances, self.main_accuracy_estimator.upper()))
        best_starting_index = training_results[0].testing_performances.first_used_index

        training_data = ingested_data.copy().loc[best_starting_index:]

        training_data.iloc[:, 0] = self.transformation.apply(training_data.iloc[:, 0])

        self.train(training_data.copy(), extra_regressors)

        future_df = pd.DataFrame(index=pd.date_range(freq=self.freq,
                                                     start=training_data.index.values[0],
                                                     periods=len(training_data) + self.prediction_lags),
                                 columns=["yhat"], dtype=training_data.iloc[:, 0].dtype)

        forecast = self.predict(future_df, extra_regressors)
        forecast.loc[:, 'yhat'] = self.transformation.inverse(forecast['yhat'])

        try:
            forecast.loc[:, 'yhat_lower'] = self.transformation.inverse(forecast['yhat_lower'])
            forecast.loc[:, 'yhat_upper'] = self.transformation.inverse(forecast['yhat_upper'])
        except:
            pass

        return forecast

    def launch_model(self, ingested_data: DataFrame, extra_regressors: DataFrame = None, max_threads: int = 1):
        """
        Train the model on `ingested_data` and returns a `ModelResult` object.
        This function is at the highest possible level of abstraction to train a model on a time-series.

        Parameters
        ----------
        ingested_data : DataFrame
            DataFrame containing the historical time series value; it will be split in training and test parts.
        extra_regressors : DataFrame, optional, default None
            Additional time-series to passed to `train` in order to improve the performances.
        max_threads : int, optional, default 1
            Maximum number of threads to use in the training phase.

        Returns
        -------
        model_result : ModelResult
            `ModelResult` containing the results of the model, trained on ingested_data.

        Examples
        --------
        First, create some training data:

        >>> dates = pd.date_range('2000-01-01', periods=30)  # Last index is 2000-01-30
        >>> ds = pd.DatetimeIndex(dates, freq="D")
        >>> a = np.arange(30, 60)
        >>> timeseries_dataframe = DataFrame(data={"a": a}, index=ds)

        Create the model, with default values:
        >>> param_config = {}
        >>> predictor = FBProphetModel(params=param_config)

        Launch the model:
        >>> model_output = predictor.launch_model(timeseries_dataframe)

        The result is an object of class `ModelResult`: it contains the best prediction...
        >>> model_output.best_prediction['yhat']
        ds
        2000-01-22    51.0
        2000-01-23    52.0
        2000-01-24    53.0
        2000-01-25    54.0
        2000-01-26    55.0
        2000-01-27    56.0
        2000-01-28    57.0
        2000-01-29    58.0
        2000-01-30    59.0
        2000-01-31    60.0
        2000-02-01    61.0
        2000-02-02    62.0
        2000-02-03    63.0
        2000-02-04    64.0
        2000-02-05    65.0
        2000-02-06    66.0
        2000-02-07    67.0
        2000-02-08    68.0
        2000-02-09    69.0
        Name: yhat, dtype: float64

        As well as the `characteristic` dictionary, which contains useful information on the model:
        >>> model_output.characteristics
        {'name': 'FBProphet', 'delta_training_percentage': 20, 'delta_training_values': 6, 'test_values': 3,
        'transformation': <timexseries.data_prediction.transformation.Identity object at 0x7f29214b3a00>}

        The `results` attribute, instead, contains the results of the training on each of the tested sub-training-sets.
        >>> model_output.results
        [<timexseries.data_prediction.models.predictor.SingleResult at 0x7f28679e4a90>,
         <timexseries.data_prediction.models.predictor.SingleResult at 0x7f28679e48b0>,
         <timexseries.data_prediction.models.predictor.SingleResult at 0x7f286793a730>,
         <timexseries.data_prediction.models.predictor.SingleResult at 0x7f28679e4c10>,
         <timexseries.data_prediction.models.predictor.SingleResult at 0x7f2875cd2490>]

        Each `SingleResult` has an attribute `prediction`, which contains the prediction
        on the validation set (in this case, composed by the last 3 values of `timeseries_dataframe`) and
        `testing_performances` which recaps the performance, in terms of MAE, MSE, etc. of that `SingleResult` on the
        validation set.
        """
        model_characteristics = self.model_characteristics

        self.delta_training_values = int(round(len(ingested_data) * self.delta_training_percentage / 100))

        if self.test_values == -1:
            self.test_values = int(round(len(ingested_data) * (self.test_percentage / 100)))

        self.freq = pd.infer_freq(ingested_data.index)

        # We need to pass ingested data both to compute_training and compute_best_prediction, so better use copy()
        # because, otherwise, we may have side effects.
        train_ts = ingested_data.copy().iloc[:-self.test_values]
        test_ts = ingested_data.copy().iloc[-self.test_values:]

        train_ts.iloc[:, 0] = self.transformation.apply(train_ts.iloc[:, 0])

        model_training_results = self._compute_trainings(train_ts, test_ts, extra_regressors, max_threads)

        best_prediction = self._compute_best_prediction(ingested_data, model_training_results, extra_regressors)

        if extra_regressors is not None:
            model_characteristics["extra_regressors"] = ', '.join([*extra_regressors.columns])

        model_characteristics["name"] = self.name
        model_characteristics["delta_training_percentage"] = self.delta_training_percentage
        model_characteristics["delta_training_values"] = self.delta_training_values
        model_characteristics["test_values"] = self.test_values
        model_characteristics["transformation"] = self.transformation

        return ModelResult(results=model_training_results, characteristics=model_characteristics,
                           best_prediction=best_prediction)
