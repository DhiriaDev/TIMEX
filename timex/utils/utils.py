from pandas import DataFrame

from timex.scenario.scenario import Scenario


def prepare_extra_regressor(scenario: Scenario, testing_performance_target: str = 'MAE') -> DataFrame:
    """
    This function receives a Scenario object which includes a prediction for a time-series and indications on the
    prediction errors, along with the entire ingested dataset.
    Then, the best possible prediction (w.r.t a specific indicator, i.e MAE) is taken and appended to the original
    time-series, in order to obtain a DataFrame with the original time series and the best possible prediction.

    The resulting DataFrame is returned.

    Parameters
    ----------
    testing_performance_target : str
    Testing performance indicator to use in order to select the best forecast. Default MAE.

    scenario : Scenario
    Scenario from which an extra-regressor should be extracted.

    Returns
    -------
    df : DataFrame
    DataFrame with the length of the original time-series + prediction lags.
    """
    name = scenario.scenario_data.columns[0]
    model_results = scenario.models[0].results
    model_results.sort(key=lambda x: getattr(x.testing_performances, testing_performance_target.upper()))

    original_ts = scenario.ingested_data[[name]]
    f = model_results[0].prediction.loc[:, ['yhat']]
    f.rename(columns={'yhat': name}, inplace=True)

    best_entire_forecast = original_ts.combine_first(f)

    return best_entire_forecast
