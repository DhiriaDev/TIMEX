import pickle, base64

import logging
log = logging.getLogger(__name__)

import sys
sys.path.append('/home/fpuoti/GIT/TimexDocker/')

from utils import *

from data_ingestion_server.data_ingestion import ingest_timeseries




kafka_address = '0.0.0.0:9092'
chunk_size = 999500  # in bytes


def ingestion_work(self):

    log.info('Data ingestion worker spawned. Waiting for data')

    data_ingestion_topic = 'data_ingestion_' + \
        self.param_config['activity_title']

    kafka_address = self.producer_config['bootstrap.servers']
    client_id = self.consumer_config['client.id']

    dataset = receive_data(
        topic=data_ingestion_topic, kafka_address = kafka_address, cons_id = client_id, consumer=self.consumer)

    
    log.info('Data received. Starting the job')

    prediction_topic = 'prediction_' + self.param_config['activity_title']
    create_topics(kafka_address=kafka_address, prod_id=client_id,
                  topics=[prediction_topic], broker_offset=1)

    dataset = ingest_timeseries(self.param_config, dataset)

    dataset = pickle.dumps(dataset)
    dataset = (base64.b64encode(dataset)).decode('utf-8')
    chunks = prepare_chunks(dataset, chunk_size)

    send_data(topic = prediction_topic, chunks = chunks, 
              file_name='dataset_' + self.param_config['activity_title'],
              prod_id = client_id, producer = self.producer)


ingestion_watcher_conf = {
    "bootstrap.servers": kafka_address,
    "client.id": 'watcher_ingestion',
    "group.id": 'watcher_ingestion',
    "max.in.flight.requests.per.connection": 1,
    "auto.offset.reset": 'earliest'
}

if __name__ == '__main__':
    ingestion_watcher = Watcher(
        config_dict=ingestion_watcher_conf, works_to_do=[ingestion_work])


    ingestion_watcher.listen_on_control(control_topic='control_topic')




