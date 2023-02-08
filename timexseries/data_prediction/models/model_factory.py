from timexseries.data_prediction import PredictionModel
from timexseries.data_prediction.models.ets import ETS
from timexseries.data_prediction.models.mockup import MockUp
from timexseries.data_prediction.models.seasonal_persistence import SeasonalPersistenceModel


def model_factory(model_class: str) -> PredictionModel:
    """
    Given the name of the model, return the corresponding PredictionModel.

    Parameters
    ----------
    model_class : str
        Model type, e.g. "fbprophet"

    Returns
    -------
    PredictionModel
        Prediction model of the class specified in `model_class`.

    Examples
    --------
    >>> model = model_factory("fbprophet")
    >>> print(type(model))
    <class 'timexseries.data_prediction.models.prophet_predictor.FBProphetModel'>
    """
    model_class = model_class.lower()

    if model_class == "seasonal_persistence" or model_class == "seasonal_naive":
        return SeasonalPersistenceModel()
    if model_class == "exponentialsmoothing" or model_class == "exponential_smoothing" or model_class == "ets":
        return ETS()
    if model_class == "mockup":
        return MockUp()
