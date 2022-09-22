from data_ingestion_server.data_ingestion import *
from utils import *
import json, pickle, base64


kafka_address = '0.0.0.0:9092'
chunk_size=999500 # in bytes


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
        "models": "fbprophet",
        "main_accuracy_estimator": "rmse"
    },
    "visualization_parameters": {
        "xcorr_graph_threshold": 0.8,
        "box_plot_frequency": "1W"
    }
}


def ingestion_work(self):

    data_ingestion_topic = 'data_ingestion_' + \
        self.param_config['activity_title']
    dataset = receive_data(
        topic=data_ingestion_topic, consumer=self.consumer)

    kafka_address = self.consumer_config['bootstrap.servers']
    prod_id = self.consumer_config['client.id']
    prediction_topic = 'prediction_' + self.param_config['activity_title']
    create_topics(kafka_address=kafka_address, prod_id=prod_id,
                  topics=[prediction_topic], broker_offset=1)

    dataset = ingest_timeseries(param_config, dataset)

    dataset = pickle.dumps(dataset)
    dataset = (base64.b64encode(dataset)).decode('utf-8')

    dataset_size = len(dataset)
    chunks_number =ceil(float(dataset_size) / float(chunk_size))
    chunks = []
    for i in range(0, chunks_number, chunk_size):
        headers = {"prod_id": str(prod_id),
                    "chunk_id": str(i),
                    "chunks_number": str(chunks_number),
                    "file_name": 'dataset_'+ self.param_config['activity_title']}

        chunks.append({"headers": headers, "data": dataset[i * chunk_size : (i+1) *chunk_size]})

    assert(len(chunks) != 0)

    send_data(prediction_topic,chunks,self.producer)



job_producer = JobProducer(prod_id=0, kafka_address=kafka_address)
job_producer.start_job(param_config, './data_to_send/BitcoinPrice.csv')

watcher_conf = {
    "bootstrap.servers": kafka_address,
    "client.id": 'watcher_ingestion',
    "group.id": 'watcher_ingestion',
    "max.in.flight.requests.per.connection": 1,
    "auto.offset.reset": 'earliest'
}


ingestion_watcher = Watcher(
    config_dict=watcher_conf, works_to_do=[ingestion_work])
ingestion_watcher.listen_on_control(control_topic='control_topic')
