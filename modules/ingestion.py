import sys
import uuid

sys.path.append('/')
sys.path.append('../')  # For local tests.

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

    # TODO: optimize topic deletion
    delete_topics(self.consumer_config, [data_ingestion_topic])


def start_worker_from_go(kafka_address: str, message: str):
    log.info('Control message arrived')

    worker_id = f"ingestion_worker_{str(uuid.uuid4())[:8]}"

    works_to_do = [ingestion_work]

    # ---- SPAWNING THE WORKER FOR THE JOB -----
    with open(base_config_path, "r") as f:
        config = json.load(f)

    worker_producer_config = config["base"].copy()
    worker_producer_config.update(config["producer"])
    worker_producer_config['bootstrap.servers'] = kafka_address
    worker_producer_config['client.id'] = worker_id

    worker_consumer_config = config["base"].copy()
    worker_consumer_config.update(config["consumer"])
    worker_consumer_config['bootstrap.servers'] = kafka_address
    worker_consumer_config['client.id'] = worker_id
    worker_consumer_config['group.id'] = 'ingestion_group'

    param_config = json.loads(message)['param_config']

    # TO_DO: check if the cons_id can be substituted by the job_name
    worker = Worker(consumer_config=worker_consumer_config,
                    producer_config=worker_producer_config,
                    works_to_do=works_to_do,
                    param_config=param_config)

    activity_title = param_config['activity_title']
    log.info(f'Spawning a worker for the job {activity_title}.')
    worker.work()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('kafka_address',
                        type=str,
                        help='single address (or list of addresses) of the form IP:port[,IP:port]')

    parser.add_argument('message',
                        type=str,
                        help='message to parse')

    args = parser.parse_args()
    if args.kafka_address is None:
        log.error('a kafka address has been not specified')
        exit(1)

    start_worker_from_go(args.kafka_address, args.message)
