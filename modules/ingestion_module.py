import sys
sys.path.append('./')

import base64, pickle
from confluent_kafka import *
import argparse

from Redpanda import *
from timexseries.data_ingestion import ingest_timeseries


import logging
log = logging.getLogger(__name__)

 


def ingestion_work(self):

    log.info('Data ingestion worker spawned. Waiting for data')

    data_ingestion_topic = 'data_ingestion_' + \
        self.param_config['activity_title']

    dataset = receive_msg(topic=data_ingestion_topic,
                           consumer_config=self.consumer_config)

    log.info('Data received. Starting the job')

    prediction_topic = 'prediction_' + self.param_config['activity_title']
    create_topics(topics=[prediction_topic],
                  client_config=self.consumer_config, broker_offset=1)

    dataset = ingest_timeseries(self.param_config, dataset)

    dataset = pickle.dumps(dataset)
    dataset = (base64.b64encode(dataset)).decode('utf-8')
    chunks = prepare_chunks(dataset, chunk_size)

    send_data_msg(topic=prediction_topic, chunks=chunks,
              file_name='dataset_' + self.param_config['activity_title'],
              producer_config=self.producer_config)


if __name__ == '__main__':

    # Initialize parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('kafka_address', 
                        type = str,
                        help='single address (or list of addresses) of the form IP:port[,IP:port]')

    args = parser.parse_args()
    if args.kafka_address is None:
        log.error('a kafka address has been not specified')
        exit(1)

    ingestion_watcher_conf = default_consumer_config.copy()
    ingestion_watcher_conf['bootstrap.servers'] = args.kafka_address
    ingestion_watcher_conf['client.id'] = 'watcher_ingestion'
    ingestion_watcher_conf['group.id'] = 'watcher_ingestion'

    ingestion_watcher = Watcher(
        config_dict=ingestion_watcher_conf, works_to_do=[ingestion_work])

    ingestion_watcher.listen_on_control(control_topic='control_topic')
