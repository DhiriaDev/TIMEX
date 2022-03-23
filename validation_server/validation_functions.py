
from utils import TimeSeriesContainer, ValidationPerformance
import logging

logger = logging.getLogger(__name__)

def validate (timeseries_containers : list[TimeSeriesContainer], param_config : dict) -> TimeSeriesContainer:
    main_accuracy_estimator = param_config["model_parameters"]["main_accuracy_estimator"]

    logger.info ('Validation Started: using {main_accuracy_estimator} as main accuracy estimator!')
    '''
    Since the prediction request are asynchronous for each requested model, now we have to merge the results of
    all the models for the requested timeseries in a single container
    '''
    timeseries_container = TimeSeriesContainer.merge_for_each_timeseries(timeseries_containers)

    models = timeseries_container.models
    models_best = {}

    for model in models:
        models[model].results.sort(key=lambda x: getattr(x.testing_performances, main_accuracy_estimator.upper()))
        models_best[model] = getattr(models[model].results[0].testing_performances, main_accuracy_estimator.upper())

    best_model_name = max(models_best, key = models_best.get)

    timeseries_container.models = { best_model_name : timeseries_container.models[best_model_name] }

    logger.info ('Validation finished: best model = {best_model_name}')

    
    return timeseries_container