import pickle, base64
import argparse
import psutil
import uuid
import sys

sys.path.append('/')
sys.path.append('.')  # For local tests.

from Redpanda import *
from timexseries.data_prediction import create_timeseries_containers
from validation_functions import validate
import multiprocessing

log = logging.getLogger(__name__)


def prediction_work(self):
    log.info('Prediction worker spawned, waiting for data')

    prediction_topic = 'prediction_' + self.param_config['activity_title']

    # The dataset after the ingestion process will come pickle encoded
    # to not loose all the attributes and information of a pandas data frame
    dataset = pickle.loads(
        base64.b64decode(
            receive_msg(topic=prediction_topic, consumer_config=self.consumer_config)
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

    result_topic = 'result_' + self.param_config['activity_title']
    create_topics(topics=[result_topic], client_config=self.consumer_config, broker_offset=1)

    # PREDICTION / INFERENCE FUNCTION
    timeseries_containers = create_timeseries_containers(dataset, self.param_config)

    # VALIDATION FUNCTION
    ts_json = json.dumps(
        validate(timeseries_containers, self.param_config)
    )

    chunks = prepare_chunks(str(ts_json), chunk_size)

    send_data_msg(topic=result_topic, chunks=chunks,
                  file_name='prediction_' + self.param_config['activity_title'],
                  producer_config=self.producer_config)

    multiprocessing.active_children()
    current_proc = psutil.Process()
    subproc = set([p.pid for p in current_proc.children(recursive=True)])
    for subproc in subproc:
        psutil.Process(subproc).terminate()

    # TODO: optimize topic deletion
    delete_topics(self.consumer_config, [prediction_topic])

def start_worker_from_go(kafka_address: str, message: str):
    log.info('Control message arrived')

    worker_id = f"prediction_worker_{str(uuid.uuid4())[:8]}"

    works_to_do = [prediction_work]

    # ---- SPAWNING THE WORKER FOR THE JOB -----
    worker_producer_config = config["base"].copy()
    worker_producer_config.update(config["producer"])
    worker_producer_config['bootstrap.servers'] = kafka_address
    worker_producer_config['client.id'] = worker_id

    worker_consumer_config = config["base"].copy()
    worker_consumer_config.update(config["consumer"])
    worker_consumer_config['bootstrap.servers'] = kafka_address
    worker_consumer_config['client.id'] = worker_id
    worker_consumer_config['group.id'] = 'predictor_group'

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
    # TO USE WITH GO WATCHER
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
