import sys
import logging
sys.path.append('./')
log = logging.getLogger(__name__)

from redpanda_modules import *

kafka_address = 'zk1.dhiria.com:9092,zk2.dhiria.com:9092,zk4.dhiria.com'


param_config = {
    "activity_title": "Bitcoin price forecasting",
    "verbose": "INFO",
    "input_parameters": {
        "columns_to_load_from_url": "Date,Close",
        "source_data_url" : "",
        "datetime_column_name": "Date",
        "index_column_name": "Date",
        "frequency": "D"
    },
    "model_parameters": {
        "validation_values": 15,
        "delta_training_percentage": 100,
        "forecast_horizon": 10,
        "possible_transformations": "none,log_modified",
        "models": "fbprophet",
        "main_accuracy_estimator": "mse"
    },
    "historical_prediction_parameters": {  
        # Historical predictions iterate the prediction phase in order to check the accuracy on a 
        # longer period. The best predictions for day x are computed using data available only at
        # day x-1. More on this later...
        "initial_index": "2021-02-19",  # Start the historical predictions from this day
        "save_path": "historical_predictions/"  # Save the historical predictions in this file
    }
}

if __name__ == '__main__':

    # ---- the following two lines of code simulate the behavior of a new incoming request for a job
    job_producer = JobProducer(prod_id=0, kafka_address=kafka_address)
    result_topic = job_producer.start_job(param_config, './dataset_examples/BitCoin/BitcoinPrice.csv')
    job_producer.end_job(result_topic)







