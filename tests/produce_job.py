import sys
import logging
sys.path.append('./')
log = logging.getLogger(__name__)

from redpanda_modules import JobProducer, JobReceiver

kafka_address = 'zk1.dhiria.com:9092,zk2.dhiria.com:9092,zk4.dhiria.com'


param_config ={
    "activity_title": "Electricity Load 2011-2014",
    "verbose": "INFO",
    "input_parameters": {
        "source_data_url": "https://drive.google.com/file/d/1LQY4bvdLVtOZ1vpko6USMgclKjuqaN8F/view?usp=sharing",
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

    # ---- the following two lines of code simulate the behavior of a new incoming request for a job
    job_producer = JobProducer(prod_id=0, kafka_address=kafka_address)
    result_topic = job_producer.start_job(param_config, '/home/eks-timex/shared/dhiria-shared/redpanda-tests/data_to_send/ElectricityLoad.csv')
    job_receiver = JobReceiver(cons_id=1000000, kafka_address=kafka_address)
    job_receiver.end_job(result_topic)







