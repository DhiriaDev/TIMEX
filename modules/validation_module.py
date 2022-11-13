import sys
import logging
sys.path.append('./')
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


    send_data_msg(topic = result_topic, chunks = chunks, 
              file_name='best_model' + self.param_config['activity_title'],
              producer_config=self.producer_config)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('kafka_address', 
                        type = str,
                        help='single address (or list of addresses) of the form IP:port[,IP:port]')


    args = parser.parse_args()
    if args.kafka_address is None:
        log.error('a kafka address has been not specified')
        exit(1)

    validation_watcher_conf = default_consumer_config.copy()
    validation_watcher_conf['bootstrap.servers'] = args.kafka_address
    validation_watcher_conf['client.id'] = 'watcher_validation'
    validation_watcher_conf['group.id'] = 'watcher_validation'


    validation_watcher = Watcher(
        config_dict=validation_watcher_conf, works_to_do=[validation_work])


    validation_watcher.listen_on_control(control_topic='control_topic')




