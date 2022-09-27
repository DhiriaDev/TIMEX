import sys
import logging
sys.path.append('./')
log = logging.getLogger(__name__)

import pickle, base64

from redpanda_modules import *
from validation_server.validation_functions import *
from confluent_kafka import *

kafka_address = '0.0.0.0:9092'
chunk_size = 999500  # in bytes


def validation_work(self):

    log.info('Validation worker spawned. Waiting for data')

    validation_topic = 'validation_' + \
        self.param_config['activity_title']

    kafka_address = self.producer_config['bootstrap.servers']
    client_id = self.consumer_config['client.id']

    consumer = Consumer(self.consumer_config)
    timeseries_containers = pickle.loads(
        base64.b64decode(
            receive_data(topic=validation_topic, kafka_address=kafka_address, cons_id=client_id, consumer=consumer)
        )
    )


    

    log.info('Results received. Starting the validation')

    result_topic = 'result_' + self.param_config['activity_title']
    create_topics(kafka_address=kafka_address, prod_id=client_id,
                  topics=[result_topic], broker_offset=1)
    try:

        if(len(timeseries_containers) > 1):
            best_model = validate(timeseries_containers, self.param_config)
        else:
            best_model = timeseries_containers[0]

    except ValueError as e:
        print (e)
        return 500

    best_model = pickle.dumps(best_model)
    best_model = base64.b64encode(best_model).decode('utf-8')

    chunks = prepare_chunks(best_model, chunk_size)

    producer = Producer(self.producer_config)
    send_data(topic = result_topic, chunks = chunks, 
              file_name='best_model' + self.param_config['activity_title'],
              prod_id = client_id, producer = producer)


validation_watcher_conf = {
    "bootstrap.servers": kafka_address,
    "client.id": 'watcher_validation',
    "group.id": 'watcher_validation',
    "max.in.flight.requests.per.connection": 1,
    "auto.offset.reset": 'earliest'
}

if __name__ == '__main__':
    validation_watcher = Watcher(
        config_dict=validation_watcher_conf, works_to_do=[validation_work])


    validation_watcher.listen_on_control(control_topic='control_topic')




