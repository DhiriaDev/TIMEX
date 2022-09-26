import subprocess

from utils import *

import logging
log = logging.getLogger(__name__)

kafka_address = '0.0.0.0:9092'

param_config = {
    "activity_title": "Bitcoin price forecasting",
    "verbose": "INFO",
    "input_parameters": {
        "columns_to_load_from_url": "Date,Close",
        "datetime_column_name": "Date",
        "index_column_name": "Date",
        "frequency": "D"
    },
    "model_parameters": {
        "test_values": 15,
        "delta_training_percentage": 15,
        "prediction_lags": 10,
        "possible_transformations": "none,log_modified",
        "models": "fbprophet,arima",
        "main_accuracy_estimator": "rmse"
    },
    "historical_prediction_parameters": {  
        # Historical predictions iterate the prediction phase in order to check the accuracy on a 
        # longer period. The best predictions for day x are computed using data available only at
        # day x-1. More on this later...
        "initial_index": "2021-02-01",  # Start the historical predictions from this day
        "save_path": "historical_predictions/"  # Save the historical predictions in this file
    },
    "visualization_parameters": {
        "xcorr_graph_threshold": 0.8,
        "box_plot_frequency": "1W"
    }
}

if __name__ == '__main__':


    subprocess.Popen(["python /home/fpuoti/GIT/TimexDocker/tests/ingestion_test.py"], shell=True)
    subprocess.Popen(["python /home/fpuoti/GIT/TimexDocker/tests/prediction_test.py"], shell = True)
    subprocess.Popen(["python /home/fpuoti/GIT/TimexDocker/tests/validation_test.py"], shell = True)

    # ---- the following two lines of code simulate the behavior of a new incoming request for a job
    job_producer = JobProducer(prod_id=0, kafka_address=kafka_address)
    job_producer.start_job(param_config, './dataset_examples/BitCoin/BitcoinPrice.csv')


