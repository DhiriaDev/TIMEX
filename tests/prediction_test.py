import sys
import logging
sys.path.append('./')
log = logging.getLogger(__name__)


import pickle, base64

from confluent_kafka import *

from redpanda_modules import *

from  timexseries.data_prediction import create_timeseries_containers


kafka_address = '0.0.0.0:9092'
chunk_size = 999500  # in bytes



def prediction_work(self):

    log.info('Prediction worker spawned, waiting for data')

    prediction_topic = 'prediction_' + self.param_config['activity_title']

    kafka_address = self.producer_config['bootstrap.servers']
    client_id = self.producer_config['client.id']

    consumer = Consumer(self.consumer_config)
    # The dataset after the ingestion process will come pickle encoded
    # to not loose all the attributes and information of a pandas data frame
    dataset = pickle.loads(
        base64.b64decode(
            receive_data(topic=prediction_topic, kafka_address=kafka_address, cons_id=client_id, consumer=consumer)
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
    create_topics(kafka_address=kafka_address, prod_id=client_id,
                  topics=[validation_topic], broker_offset=1)

    chunks = prepare_chunks(timeseries_containers, chunk_size)
    
    producer = Producer(self.producer_config)
    send_data(topic=validation_topic, chunks=chunks,
              file_name='prediction_' + self.param_config['activity_title'], 
              prod_id=client_id, producer=producer)


prediction_watcher_conf = {
    "bootstrap.servers": kafka_address,
    "client.id": 'watcher_prediction',
    "group.id": 'watcher_prediction',
    "max.in.flight.requests.per.connection": 1,
    "auto.offset.reset": 'earliest'
}

if __name__ == '__main__':

    prediction_watcher = Watcher(
        config_dict=prediction_watcher_conf, works_to_do=[prediction_work])


    prediction_watcher.listen_on_control(control_topic='control_topic')



