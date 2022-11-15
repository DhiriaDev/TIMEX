import sys
import logging
sys.path.append('./')
log = logging.getLogger(__name__)

import argparse

from Redpanda import JobProducer


param_config ={
    "activity_title": "Electricity Load 2011-2014",
    "verbose": "INFO",
    "input_parameters": {
        "source_data_url": "",
        "frequency": "15T",
        "columns_to_load_from_url": "Date,MT_001",
        "datetime_column_name": "Date"
    },
    "model_parameters": {
        "validation_values": 10,
        "delta_training_percentage": 30,
        "forecast_horizon": 10,
        "possible_transformations": "none,log_modified",
        "models": "fbprophet",
        "main_accuracy_estimator": "rmse"
    },
    "visualization_parameters": {
        "xcorr_graph_threshold": 0.8,
        "box_plot_frequency": "1W"
    }
}



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        'kafka_address', 
        type = str,
        help='single address (or list of addresses) of the form IP:port[,IP:port]'
    )                   
    parser.add_argument(
        'file_path',
        type = str,
        help = 'Path to were take the input file that is to be sent for the job. The absolute path is suggested'
    )

    args = parser.parse_args()
    kafka_address = args.kafka_address
    file_path = args.file_path

    if kafka_address is None:
        log.error('a kafka address has been not specified')
        exit(1)
    if file_path is None:
        log.error('a file path has been not specified')
        exit(1)

    job_producer = JobProducer(client_id = 0, kafka_address=kafka_address)
    job_producer.start_job(param_config, file_path)
    results = job_producer.end_job()
    print(results)








