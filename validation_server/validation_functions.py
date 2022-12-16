import logging
import pandas as pd

logger = logging.getLogger(__name__)

# TODO: diamo per scontato che i modelli abbiano tutti lo stesso nome nelle colonne. Scegliere una nomenclatura
#  standard.
def filter_prediction_field(best_pred: pd.DataFrame, column: str)-> pd.DataFrame:
    """
    Filters out the columns that are not relevant for a "non-debug" result output, when present
    """
    if 'yhat_lower' and 'yhat_upper' in best_pred.columns:
        #mux = pd.MultiIndex.from_product([[column], ['yhat', 'yhat_lower', 'yhat_upper']])
        best_pred = best_pred[['yhat', 'yhat_lower', 'yhat_upper']]
        columns = [f'{column}-yhat', f'{column}-yhat_lower', f'{column}-yhat_upper']
    elif 'yhat' in best_pred.columns:
        #mux = pd.MultiIndex.from_product([[column], ['yhat']])
        best_pred = best_pred['yhat']
        columns = [f'{column}-yhat']
    else:
        logger.info("Unexpected column label in Timeseries_container")
    df = pd.DataFrame(best_pred.values, columns=columns, index=best_pred.index)

    return df


def validate(timeseries_containers, param_config):
    main_accuracy_estimator = param_config["model_parameters"]["main_accuracy_estimator"]
    main_accuracy_estimator = 'MAE'
    df_data = pd.DataFrame()
    prediction = pd.DataFrame()
    val_err = float
    json_result = {"data": "",
                   "best_pred": {}}
    #models_config = {}

    logger.info('Validation started using %s as main accuracy estimator!', main_accuracy_estimator)

    '''
    Since the prediction request are asynchronous for each requested model, now we have to merge the results of
    all the models for the requested timeseries in a single container
    '''

    for timeseries_container in timeseries_containers:

        models = timeseries_container.models
        models_best = {}
        column = timeseries_container.timeseries_data.columns.values[0]
        df_data = pd.concat([df_data, timeseries_container.timeseries_data], ignore_index=False, axis=1)
#        models_config[column] = []

        for model in models:
            models[model].results.sort(key=lambda x: getattr(x.testing_performances, main_accuracy_estimator.upper()))
#            models_config[column].append(models[model].characteristics)
            models_best[model] = getattr(models[model].results[0].testing_performances, main_accuracy_estimator.upper())

        val_err = min(models_best.values())
        best_model = min(models_best, key=models_best.get)
        best_pred = models[best_model].best_prediction
        best_pred = filter_prediction_field(best_pred, column)
        prediction = pd.concat([prediction, best_pred], axis=1)

    data_json = df_data.to_json(orient='columns', date_format='iso')
    prediction = prediction.to_json(orient='columns', date_format='iso')

    json_result["data"] = data_json
    json_result["best_pred"]["df_pred"] = prediction
    json_result["best_pred"]["val_err"] = val_err
#    json_result["models"].append(models_config)

    logger.info('Validation finished.')

    return json_result

