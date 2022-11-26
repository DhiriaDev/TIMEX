import sys
import logging
import uuid

sys.path.append('/')
sys.path.append('../')  # For local tests.
log = logging.getLogger(__name__)

import pickle, base64
from confluent_kafka import *
import argparse

from Redpanda import *
from validation_server.validation_functions import *


def validation_work(self):

    log.info('Validation worker spawned. Waiting for data')

    validation_topic = 'validation_' + \
        self.param_config['activity_title']

    timeseries_containers = pickle.loads(
        base64.b64decode(
            receive_msg(topic=validation_topic, consumer_config=self.consumer_config)
        )
    )

    log.info('Results received. Starting the validation.')

    result_topic = 'result_' + self.param_config['activity_title']
    create_topics(topics=[result_topic], client_config=self.consumer_config, broker_offset=1)
    try:
        if(len(timeseries_containers) > 1):
            best_model = validate(timeseries_containers, self.param_config)
        else:
            best_model = timeseries_containers[0]        
            log.info('Just one model. No validation will be performed.')

    except ValueError as e:
        print (e)
        return 500

    best_model = pickle.dumps(best_model)
    best_model = base64.b64encode(best_model).decode('utf-8')

    chunks = prepare_chunks(best_model, chunk_size)

    send_data_msg(topic = result_topic, chunks=chunks,
                  file_name='best_model' + self.param_config['activity_title'],
                  producer_config=self.producer_config)

    # TODO: optimize topic deletion
    delete_topics(self.consumer_config, [validation_topic])

def start_worker_from_go(kafka_address: str, message: str):
    log.info('Control message arrived')

    worker_id = f"validation_worker_{str(uuid.uuid4())[:8]}"

    works_to_do = [validation_work]

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
    worker_consumer_config['group.id'] = 'validation_group'

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




