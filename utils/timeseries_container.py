from pandas import DataFrame
from itertools import groupby

class TimeSeriesContainer:
    """
    A TimeSeriesContainer collect all the relevant information useful to characterize a single time-series coming from
    the ingested dataset.

    Parameters
    ----------
    timeseries_data : DataFrame
        Historical time-series data, in the form of a DataFrame with a index and a single data column.
    models : dict
        Dictionary of ModelResult objects, all trained on this time-series.
    xcorr : dict
        Cross-correlation between the data of this time-series and all the other ones.
    historical_prediction : dict
        The historical prediction, i.e. the predictions computed on a rolling window on the historical data.
        This is useful to verify the performances of each model not only on the very last data, but throughout the
        history of the time-series, in a cross-validation fashion. This dictionary contains one entry for each model
        tested.
    """
    def __init__(self, timeseries_data: DataFrame, models: dict, xcorr: dict, historical_prediction: dict = None):
        self.timeseries_data = timeseries_data
        self.models = models
        self.xcorr = xcorr
        self.historical_prediction = historical_prediction

    def set_historical_prediction(self, historical_prediction):
        self.historical_prediction = historical_prediction

    @staticmethod
    def merge_for_each_timeseries(timeseries_containers : list["TimeSeriesContainer"]):
        """
        After the deployment of the validator module (which takes a list of time-series containers 
        and returns a the model with the best performance) there's no more need of this function.
        However, this function takes a list of timeseries_containers, and for each timeseries it merges the results 
        of all the models.
        The timex_manager performs multiple async requests to the prediction server, one for each requested model.
        Therefore, the predictions of each model for each timeseries will come back in different timeseries containers. 

        Parameters
        ----------
        timeseries_containers: list[TimeSeriesContainer]
            All the results returned by the timex_manager.

        Returns
        -------
        list[TimeSeriesContainer]
            List of timeseries containers, one for each timeseries.
        """

        # Function that returns the attribute used to sort and to group the containers 
        def groupKey(x):
            return x.timeseries_data.columns[0]

        def merge_routine(elements : list[TimeSeriesContainer]) -> TimeSeriesContainer :
            merged_container = elements[0]
            for i in range(1, len(elements)):
                merged_container.models.update(elements[i].models)
                if merged_container.historical_prediction is not None: 
                    merged_container.historical_prediction.update(elements[i].historical_prediction)
            return merged_container

        # itertools.groupby generates a break or new group every time the value of the key function changes.
        # Therefore, it is usually necessary to have sorted the data using the same key function.
        sorted_list = sorted(timeseries_containers, key=groupKey)

        for key, elements in groupby(sorted_list, key=groupKey):
            merged_container = merge_routine(list(elements));
        
        return merged_container
        