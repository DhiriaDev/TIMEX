import sys
import logging
sys.path.append('./')
log = logging.getLogger(__name__)

from confluent_kafka import *
import pickle, base64
import argparse

from redpanda_modules import *
from timexseries.data_prediction import create_timeseries_containers



def prediction_work(self):

    log.info('Prediction worker spawned, waiting for data')

    prediction_topic = 'prediction_' + self.param_config['activity_title']

    # The dataset after the ingestion process will come pickle encoded
    # to not loose all the attributes and information of a pandas data frame
    dataset = pickle.loads(
        base64.b64decode(
            receive_data(topic=prediction_topic, consumer_config=self.consumer_config)
        )
    )

    log.info('Data received. Starting the job')

    if 'historical_prediction_parameters' in self.param_config:
        path = self.param_config['historical_prediction_parameters']['save_path']
        # if historical predictions non present yet, prepare folder for data saving
        # in case of request for historical predictions
        if (not os.path.exists(path)):
            cur = ""
            print('Finding the path...')
            for dir in (path.split('/')[:-1]):
                cur += dir
                if (not os.path.exists(cur)):
                    print('not present')
                    os.makedirs(cur)

    timeseries_containers = create_timeseries_containers(dataset, self.param_config)

    timeseries_containers = pickle.dumps(timeseries_containers)
    timeseries_containers = base64.b64encode(timeseries_containers).decode('utf-8')

    validation_topic = 'validation_' + self.param_config['activity_title']
    create_topics(topics=[validation_topic], client_config=self.consumer_config, broker_offset=1)

    chunks = prepare_chunks(timeseries_containers, chunk_size)

    send_data(topic=validation_topic, chunks=chunks,
              file_name='prediction_' + self.param_config['activity_title'], 
              producer_config=self.producer_config)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('kakfa_address', 
                        type = str,
                        help='single address (or list of addresses) of the form IP:port[,IP:port]')


    args = parser.parse_args()
    if args.kafka_address is None:
        log.error('a kafka address has been not specified')
        exit(1)

    prediction_watcher_conf = {
        "bootstrap.servers": args.kafka_address,
        "client.id": 'watcher_prediction',
        "group.id": 'watcher_prediction',
        "max.in.flight.requests.per.connection": 1,
        "auto.offset.reset": 'earliest'
    }

    prediction_watcher = Watcher(
        config_dict=prediction_watcher_conf, works_to_do=[prediction_work])


    prediction_watcher.listen_on_control(control_topic='control_topic')



