import src.data_ingestion.data_ingestion
import src.data_preparation.data_preparation
import src.data_visualization.data_visualization
from src import timex as tx
import json

# data,stato,ricoverati_con_sintomi,terapia_intensiva,totale_ospedalizzati,isolamento_domiciliare,totale_positivi,variazione_totale_positivi,nuovi_positivi,dimessi_guariti,deceduti,casi_da_sospetto_diagnostico,casi_da_screening,totale_casi,tamponi,casi_testati,note

# CHANGE HERE TO CHANGE EXAMPLE
# Can be: "covid19", "airlines"
example = "airlines"

if example == "airlines":
    # PARAMETERS
    param_file_nameJSON = 'configuration_test_airlines.json'

    # Load parameters from config file.
    with open(param_file_nameJSON) as json_file:  # opening the config_file_name
        param_config = json.load(json_file)  # loading the json

    if param_config["verbose"] == 'yes':
        print('data_ingestion: ' + 'json file loading completed!')

    # data ingestion
    print('-> INGESTION')
    df = src.data_ingestion.data_ingestion.data_ingestion(param_config)   # ingestion of data

    print('-> SELECTION')
    df = src.data_preparation.data_preparation.data_selection(df, param_config)

    print('-> DESCRIPTION')
    src.data_visualization.data_visualization.data_description(df, param_config)

if example == "covid19":
    # PARAMETERS
    param_file_nameJSON = 'configuration_test_covid19.json'

    # Load parameters from config file.
    with open(param_file_nameJSON) as json_file:  # opening the config_file_name
        param_config = json.load(json_file)  # loading the json

    if param_config["verbose"] == 'yes':
        print('data_ingestion: ' + 'json file loading completed!')

    # data ingestion
    print('-> INGESTION')
    df = src.data_ingestion.data_ingestion.data_ingestion(param_config)  # ingestion of data

    print('-> SELECTION')
    df = src.data_preparation.data_preparation.data_selection(df, param_config)

    print('-> ADD DIFF COLUMN')
    df = src.data_preparation.data_preparation.add_diff_column(df, df.columns[1], 'incremento_column')

    print('-> DESCRIPTION')
    src.data_visualization.data_visualization.data_description(df, param_config)

    # print('-> DESCRIPTION')
    # src.data_visualization.data_visualization.data_frame_visualization_plotly(df, param_config, visualization_type="hist")
    # src.data_visualization.data_visualization.data_frame_visualization_plotly(df, param_config, visualization_type="line", mode="stacked")