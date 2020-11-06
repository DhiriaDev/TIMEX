import timex.data_ingestion.data_ingestion
import timex.data_preparation.data_preparation
import timex.data_visualization.data_visualization
import json

from timex.data_prediction.arima_predictor import ARIMA
from timex.data_prediction.prophet_predictor import FBProphet

# data,stato,ricoverati_con_sintomi,terapia_intensiva,totale_ospedalizzati,isolamento_domiciliare,totale_positivi,variazione_totale_positivi,nuovi_positivi,dimessi_guariti,deceduti,casi_da_sospetto_diagnostico,casi_da_screening,totale_casi,tamponi,casi_testati,note

# CHANGE HERE TO CHANGE EXAMPLE
# Can be: "covid19italy", "covid19switzerland", "airlines", "temperatures"
from timex.scenario.scenario import Scenario


def launch():
    example = "covid19italy"
    # example = "airlines"
    # example = "covid19switzerland"
    # example = "temperatures"

    if example == "covid19italy":
        param_file_nameJSON = 'configuration_test_covid19italy.json'
    elif example == "airlines":
        param_file_nameJSON = 'configuration_test_airlines.json'
    elif example == "covid19switzerland":
        param_file_nameJSON = 'configuration_test_covid19switzerland.json'
    elif example == "temperatures":
        param_file_nameJSON = 'configuration_test_daily_min_temperatures.json'
    else:
        exit()

    # Load parameters from config file.
    with open(param_file_nameJSON) as json_file:  # opening the config_file_name
        param_config = json.load(json_file)  # loading the json

    if param_config["verbose"] == 'yes':
        print('data_ingestion: ' + 'json file loading completed!')

    # data ingestion
    print('-> INGESTION')
    ingested_data = timex.data_ingestion.data_ingestion.data_ingestion(param_config)  # ingestion of data

    # data selection
    print('-> SELECTION')
    ingested_data = timex.data_preparation.data_preparation.data_selection(ingested_data, param_config)

    if "add_diff_column" in param_config["input_parameters"]:
        print('-> ADD DIFF COLUMN')
        target = param_config["input_parameters"]["add_diff_column"]
        name = target + "_diff"
        ingested_data = timex.data_preparation.data_preparation.add_diff_column(ingested_data, target, name, "yes")

    columns = ingested_data.columns
    scenarios = []
    for col in columns:
        scenario_data = ingested_data[[col]]
        model_results = []

        print('-> PREDICTION FOR ' + str(col))
        predictor = FBProphet(param_config)
        prophet_result = predictor.launch_model(scenario_data.copy())
        model_results.append(prophet_result)
        #
        predictor = ARIMA(param_config)
        arima_result = predictor.launch_model(scenario_data.copy())
        model_results.append(arima_result)

        scenarios.append(
            Scenario(scenario_data, model_results)
        )

    print('-> DESCRIPTION')
    timex.data_visualization.data_visualization.data_description_new(scenarios, param_config)


if __name__ == '__main__':
    launch()
