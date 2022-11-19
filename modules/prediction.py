import pickle, base64
import argparse
import psutil
import uuid
import sys

sys.path.append('/')
sys.path.append('../')  # For local tests.

from Redpanda import *
from timexseries.data_prediction import create_timeseries_containers
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

    timeseries_containers = create_timeseries_containers(dataset, self.param_config)

    timeseries_containers = pickle.dumps(timeseries_containers)
    timeseries_containers = base64.b64encode(timeseries_containers).decode('utf-8')

    validation_topic = 'validation_' + self.param_config['activity_title']
    create_topics(topics=[validation_topic], client_config=self.consumer_config, broker_offset=1)

    chunks = prepare_chunks(timeseries_containers, chunk_size)

    send_data_msg(topic=validation_topic, chunks=chunks,
                  file_name='prediction_' + self.param_config['activity_title'],
                  producer_config=self.producer_config)

    multiprocessing.active_children()
    current_proc = psutil.Process()
    subproc = set([p.pid for p in current_proc.children(recursive=True)])
    for subproc in subproc:
        psutil.Process(subproc).terminate()


def start_worker_from_go(kafka_address: str, message: str):
    log.info('Control message arrived')

    worker_id = f"prediction_worker_{str(uuid.uuid4())[:8]}"

    works_to_do = [prediction_work]

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
    worker_consumer_config['group.id'] = 'predictor_group'

    param_config = json.loads(message)['param_config']

    # TO_DO: check if the cons_id can be substituted by the job_name
    worker = Worker(consumer_config=worker_consumer_config,
                    producer_config=worker_producer_config,
                    works_to_do=works_to_do,
                    param_config=param_config)

    activity_title = param_config['activity_title']
    log.info(f'Spawning a worker for the job {activity_title}.')
    p = Process(target=worker.work)
    p.start()
    p.join()


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

    # TO USE WITH PYTHON WATCHER
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument('kafka_address',
    #                     type=str,
    #                     help='single address (or list of addresses) of the form IP:port[,IP:port]')
    #
    # args = parser.parse_args()
    # if args.kafka_address is None:
    #     log.error('a kafka address has been not specified')
    #     exit(1)
    #
    # with open(base_config_path, "r") as f:
    #     config = json.load(f)
    #
    # prediction_watcher_config = config["base"].copy()
    # prediction_watcher_config['bootstrap.servers'] = args.kafka_address
    # prediction_watcher_config['client.id'] = 'watcher_prediction'
    # prediction_watcher_config['group.id'] = 'watcher_prediction'
    #
    # prediction_watcher = Watcher(
    #     config_dict=prediction_watcher_config, works_to_do=[prediction_work])
    #
    # prediction_watcher.listen_on_control(control_topic='control_topic')
