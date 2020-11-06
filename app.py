import json
import time

import dash
import dash_html_components as html

from timex.data_ingestion import data_ingestion
from timex.data_prediction.arima_predictor import ARIMA
from timex.data_prediction.prophet_predictor import FBProphet
from timex.data_preparation import data_preparation
from timex.data_visualization.data_visualization import line_plot, histogram_plot, box_plot, autocorrelation_plot, \
    prediction_plot, performance_plot
from timex.scenario.scenario import Scenario

scenarios = []


def update_scenarios_every():
    period = 60 * 60 * 24
    while True:
        update_scenarios()
        print("data updated")
        time.sleep(secs=period)


def update_scenarios():
    global scenarios

    local_scenarios = []
    if param_config["verbose"] == 'yes':
        print('data_ingestion: ' + 'json file loading completed!')

    # data ingestion
    print('-> INGESTION')
    ingested_data = data_ingestion.data_ingestion(param_config)  # ingestion of data

    # data selection
    print('-> SELECTION')
    ingested_data = data_preparation.data_selection(ingested_data, param_config)

    if "add_diff_column" in param_config["input_parameters"]:
        print('-> ADD DIFF COLUMN')
        target = param_config["input_parameters"]["add_diff_column"]
        name = target + "_diff"
        ingested_data = data_preparation.add_diff_column(ingested_data, target, name, "yes")

    columns = ingested_data.columns
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

        local_scenarios.append(
            Scenario(scenario_data, model_results)
        )

        scenarios = local_scenarios


def make_layout():
    children = []
    children.append(
        html.Div(
            html.P(['The data contained in this page is an example of the capabilities of TIMEX, '
                    'a work-in-progress framework for automatic time series analysis.', html.Br(),
                    'The forecasts provided in this page are relative to the evolution of Covid-19 in Italy, '
                    'built with the ', html.A("data", href='https://github.com/pcm-dpc/COVID-19', target="_blank"),
                    ' provided by Italian Protezione Civile.', html.Br(), html.Br(),
                    'The information on this site is not intended or implied to be a substitute for '
                    'professional medical advice, diagnosis or treatment. All content, including text, '
                    'graphics, images and information, contained on or available through this web site is for '
                    'general information purposes only.', html.Br(), 'We make no representation and assume no '
                    'responsibility for the accuracy of information contained on or available through this web '
                    'site, and such information is subject to change without notice. You are encouraged to '
                    'confirm any information obtained from or through this web site with other sources.', html.Br(),
                    html.Br(),
                    'For suggestions and questions contact us at manuel.roveri (at) polimi.it or alessandro.falcetta '
                    '(at) mail.polimi.it '
                    ])))

    for s in scenarios:
        ingested_data = s.ingested_data
        models = s.models

        # Data visualization with plots
        children.extend([
            html.H1(children=param_config["activity_title"]),
            html.H2(children=ingested_data.columns[0] + " analysis"),
            html.H3("Data visualization"),
            line_plot(ingested_data),
            histogram_plot(ingested_data),
            box_plot(ingested_data, visualization_parameters["box_plot_frequency"]),
            autocorrelation_plot(ingested_data)
        ])

        # Prediction results
        children.append(
            html.H3("Training & Prediction results"),
        )

        for model in models:
            model_results = model.results
            model_characteristic = model.characteristics

            test_percentage = model_parameters['test_percentage']
            test_values = int(round(len(ingested_data) * (test_percentage / 100)))

            main_accuracy_estimator = model_parameters["main_accuracy_estimator"]
            model_results.sort(key=lambda x: getattr(x.testing_performances, main_accuracy_estimator.upper()))

            best_prediction = model_results[0].prediction
            testing_performances = [x.testing_performances for x in model_results]

            children.extend([
                html.Div("Model characteristics:"),
                html.Ul([html.Li(key + ": " + str(model_characteristic[key])) for key in model_characteristic]),
                # html.Div("Testing performance:"),
                # html.Ul([html.Li(key + ": " + str(testing_performances[key])) for key in testing_performances]),
                prediction_plot(ingested_data, best_prediction, test_values),
                performance_plot(ingested_data, best_prediction, testing_performances, test_values),
            ])
        return html.Div(children)

        # EXTRA
        # Warning: this will plot every model result, with every training set used!
        # children.extend(plot_every_prediction(ingested_data, model_results, main_accuracy_estimator, test_values))


# Initialize Dash app.
app = dash.Dash(__name__)
server = app.server

example = "covid19italy"

if example == "covid19italy":
    param_file_nameJSON = 'timex/configuration_test_covid19italy.json'
elif example == "airlines":
    param_file_nameJSON = 'timex/configuration_test_airlines.json'
elif example == "covid19switzerland":
    param_file_nameJSON = 'timex/configuration_test_covid19switzerland.json'
elif example == "temperatures":
    param_file_nameJSON = 'timex/configuration_test_daily_min_temperatures.json'
else:
    exit()

# Load parameters from config file.
with open(param_file_nameJSON) as json_file:  # opening the config_file_name
    param_config = json.load(json_file)  # loading the json

visualization_parameters = param_config["visualization_parameters"]
model_parameters = param_config["model_parameters"]

# Get initial data
update_scenarios()

print("Serving the layout...")
app.layout = make_layout

# schedule.every().day.at("18:30").do(update_scenarios, 'It is 18:30')
#
#
# def worker():
#     while True:
#         schedule.run_pending()
#         time.sleep(60)  # wait one minute
#
#
# executor = ThreadPoolExecutor(max_workers=1)
# executor.submit(worker)

if __name__ == '__main__':
    app.run_server()
