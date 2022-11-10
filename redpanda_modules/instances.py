from ast import ExceptHandler
from confluent_kafka.admin import *
from confluent_kafka import *
from multiprocessing import *

import sys, time
import json, hashlib

from .constants import *
from .utils import *

import logging

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger(__name__)


class Watcher(object):

    def __init__(self, config_dict: dict, works_to_do: list):
        self.consumer_config = config_dict
        self.works_to_do = works_to_do

    def listen_on_control(self, control_topic: str):
        receive_msg(topic=control_topic, consumer_config=self.consumer_config, works_to_do=self.works_to_do)


class Worker(object):

    def __init__(self, consumer_config: dict, producer_config: dict, works_to_do: list, param_config: dict):
        self.consumer_config = consumer_config
        self.producer_config = producer_config

        self.param_config = param_config
        self.works_to_do = works_to_do

    def work(self):
        for job in self.works_to_do:
            job(self)


class JobReceiver(object):
    def __init__(self, cons_id, kafka_address):
        self.consumer_config = default_consumer_config.copy()
        self.consumer_config['bootstrap.servers'] = kafka_address
        self.consumer_config['client.id'] = str(cons_id)
        self.consumer_config['group.id'] = 'end_job'

    def end_job(self, result_topic):
        receive_msg(result_topic, self.consumer_config)

class JobProducer(object):
    def __init__(self, prod_id, kafka_address):

        self.producer_config = default_producer_config.copy()
        self.producer_config['bootstrap.servers'] = kafka_address
        self.producer_config['client.id'] = str(prod_id)

    def start_job(self, param_config: dict, file_path: str, chunk_size=999500):

        if file_path is None:
            print('Please insert a valid file_path for the file to send')
            exit(1)

        # JOB NAME
        job_uuid = hashlib.md5(
            string=(file_path + param_config['activity_title']).encode('utf-8')
        ).hexdigest()

        param_config['activity_title'] = job_uuid

        if "historical_prediction_parameters" in param_config:
            param_config['historical_prediction_parameters']['save_path'] = 'historical_predictions/' + str(
                job_uuid) + '.pkl'

        # Notify all the watchers subscribed to the control topic
        send_control_msg(self.producer_config, param_config, control_topic='control_topic')

        fts = File(path=file_path, chunk_size=chunk_size)

        # TODO: decidere se creare già da qui il result topic
        data_topic = 'data_ingestion_' + param_config['activity_title']
        result_topic = 'result_' + param_config['activity_title']
        create_topics(topics=[data_topic], client_config=self.producer_config, broker_offset=1)
        create_topics(topics=[result_topic], client_config=self.producer_config, broker_offset=1)

        chunks = read_in_chunks(file_to_read=fts, CHUNK_SIZE=fts.get_chunk_size())

        send_data_msg(topic=data_topic, chunks=chunks,
                      file_name=fts.get_name(), producer_config=self.producer_config)

        return result_topic



def send_control_msg(producer_config: dict, param_config: dict, control_topic):
    log.info(f'Sending {MessageType.control_message.name}')

    producer = Producer(producer_config)

    headers = {'prod_id': producer_config['client.id'],
               'type': str(MessageType.control_message.value)}

    data = {'param_config': param_config}

    producer.produce(topic=control_topic,
                     value=json.dumps(data).encode('utf-8'),
                     headers=headers)
    producer.flush()
    log.info(f'{MessageType.control_message.name} successfully sent')


def send_data_msg(topic: str, chunks: list, file_name: str, producer_config: dict):
    log.info(f'Sending {MessageType.data_message.name}: {file_name}')

    chunks_number = len(chunks)
    producer = Producer(producer_config)
    for i in range(0, chunks_number):
        msg_header = {
            'type': str(MessageType.data_message.value),
            "prod_id": producer_config['client.id'],
            "chunk_id": str(i),
            "chunks_number": str(chunks_number),
            "file_name": file_name}

        producer.produce(topic=topic, value=chunks[i], headers=msg_header)

    producer.flush()
    log.info(f'{MessageType.data_message.name} successfully sent')


def receive_msg(topic: str, consumer_config: dict, works_to_do=None):
    '''
    This utility is used to receive messages on topics.

    :param topic: the topic the client will subscribe to
    :param consumer_config: this is the configuration the client(s) will have
    :param works_to_do: optional. It is used by a Watcher instance when spawning new workers

    :returns: None if record_list is None, else the a string of bytes representing the data received.

    '''

    admin_config = base_config.copy()
    admin_config['bootstrap.servers'] = consumer_config['bootstrap.servers']
    admin_config['client.id'] = consumer_config['client.id']
    admin_client = AdminClient(admin_config)

    consumer = Consumer(consumer_config)

    try:
        while topic not in admin_client.list_topics().topics:
            log.debug(f'the topic {topic} does not exist: waiting..')
            time.sleep(5)

        consumer.subscribe([topic])
        log.info(f'Listening on topic: {topic}')

        worker_id = 0  # this variable will be used to spawn new workers if needed
        record_list = None

        running = True
        while running:

            msg = consumer.poll(timeout=1)

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition event
                    sys.stderr.write(
                        "%% %s [%d] reached end at offset %d\n"
                        % (msg.topic(), msg.partition(), msg.offset())
                    )
                elif msg.error():
                    print(msg.error())

            else:
                parsed_msg = parse_msg(msg)
                msg_type = MessageType(int(parsed_msg['type'])).name

                if msg_type == 'control_message':

                    log.info('Control message arrived')
                    assert (works_to_do is not None)

                    # ---- SPAWNING THE WORKER FOR THE JOB -----

                    kafka_address = consumer_config['bootstrap.servers']
                    worker_client_id = consumer_config['client.id'] + '_worker' + str(worker_id)
                    worker_group_id = consumer_config['group.id'] + '_worker' + str(worker_id)

                    worker_consumer_config = default_consumer_config.copy()
                    worker_consumer_config["bootstrap.servers"] = kafka_address
                    worker_consumer_config['client.id'] = worker_client_id
                    worker_consumer_config['group.id'] = worker_group_id

                    worker_producer_config = default_producer_config.copy()
                    worker_producer_config['bootstrap.servers'] = kafka_address
                    worker_producer_config['client.id'] = worker_client_id

                    param_config = json.loads(
                        parsed_msg['data'].decode('utf-8'))['param_config']

                    # TO_DO: check if the cons_id can be substituted by the job_name
                    worker = Worker(consumer_config=worker_consumer_config,
                                    producer_config=worker_producer_config,
                                    works_to_do=works_to_do,
                                    param_config=param_config)

                    activity_title = param_config['activity_title']
                    log.info(f'Spawning the worker n° {worker_id} for the job {activity_title}.')
                    Process(target=worker.work).start()

                    worker_id += 1

                elif msg_type == 'data_message':

                    chunk_id = int(parsed_msg["chunk_id"])
                    chunks_number = int(parsed_msg["chunks_number"])

                    if chunk_id + 1 > chunks_number:
                        sys.stderr.write("Unexpected msg chunk id")

                    if record_list is None:
                        record_list = [{} for _ in range(chunks_number)]

                    # message ordering is guaranteed even if they do not arrive in order
                    record_list[chunk_id] = parsed_msg

                    # el is a dictionary and to check if it is empty it suffices to check if its len == 0
                    still_to_receive_chunks = True in (
                        len(el) == 0 for el in record_list)

                    if not still_to_receive_chunks:
                        running = False

                else:
                    log.error(f'UnknownMessageType : {msg_type}')

                consumer.store_offsets(msg)

    except Exception as e:
        log.error(e)
    finally:
        # Close down consumer to commit final offsets.
        consumer.close()

    if record_list is not None:
        return b"".join([item["data"] for item in record_list])
    else:
        return None


def parse_msg(msg):
    decoded_msg = {}
    # msg header is a list of the type [[header_name, header_value], ...]
    for header in msg.headers():
        decoded_msg[header[0]] = header[1].decode()

    decoded_msg["data"] = msg.value()

    return decoded_msg
