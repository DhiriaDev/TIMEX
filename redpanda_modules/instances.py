from .utils import *

from confluent_kafka import Producer, Consumer
from confluent_kafka.admin import *

import json, hashlib
from multiprocessing import *

import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger(__name__)



class Watcher(object):

    def __init__(self, config_dict: dict, works_to_do: list):
        self.config = config_dict
        self.consumer = Consumer(self.config)
        self.works_to_do = works_to_do

    def listen_on_control(self, control_topic: str):
        try:
            self.consumer.subscribe([control_topic])
            logging.info('listening on control topic')
            running = True
            worker_id = 0

            while running:
                msg = self.consumer.poll(timeout=1)
                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # End of partition event
                        sys.stderr.write('%% %s [%d] reached end at offset %d\n' %
                                         (msg.topic(), msg.partition(), msg.offset()))
                    elif msg.error():
                        raise KafkaException(msg.error())
                else:
                    parsed_msg = parse_msg(msg)
                    msg_type = MessageType(int(parsed_msg['type'])).name

                    if msg_type == 'control_message':
                        log.info('Control message arrived')

                        # ------------------------------------------
                        # ---- SPAWNING THE WORKER FOR THE JOB -----
                        # ------------------------------------------

                        kafka_address = self.config['bootstrap.servers']
                        consumer_conf = {
                            "bootstrap.servers": kafka_address,
                            "client.id": str(worker_id),
                            "group.id": 'cons' + str(worker_id),
                            "max.in.flight.requests.per.connection": 1,
                            "auto.offset.reset": 'earliest'
                        }

                        producer_conf = {
                            "bootstrap.servers": kafka_address,
                            "client.id": str(worker_id),
                            "acks": 1,
                            "retries": 3,
                            "batch.size": 1,
                            "max.in.flight.requests.per.connection": 1
                        }

                        param_config = json.loads(
                            parsed_msg['data'].decode('utf-8'))['param_config']

                        # TO_DO: check if the cons_id can be substituted by the job_name
                        worker = Worker(consumer_config=consumer_conf,
                                        producer_config=producer_conf,
                                        works_to_do=self.works_to_do,
                                        param_config=param_config)

                        #Process(target=worker.work).start()
                        worker.work()

                        worker_id += 1

        finally:
            self.consumer.close()


class Worker(object):

    def __init__(self, consumer_config: dict, producer_config: dict, works_to_do: list, param_config: dict):

        self.consumer_config = consumer_config
        self.producer_config = producer_config

        self.consumer = Consumer(consumer_config)
        self.producer = Producer(producer_config)

        self.param_config = param_config
        self.works_to_do = works_to_do

    def work(self):
        for job in self.works_to_do:
            job(self)


class JobProducer(object):
    def __init__(self, prod_id, kafka_address):
        self.prod_id = prod_id
        self.kafka_address = kafka_address

    def start_job(self, param_config: dict, file_path: str, chunk_size=999500):

        if file_path is None:
            print('Please insert a valid file_path for the file to send')
            exit(1)

        # JOB NAME
        job_uuid = hashlib.md5(
            string= (file_path + param_config['activity_title']).encode('utf-8')
        ).hexdigest()
        
        param_config['activity_title'] = job_uuid

        if "historical_prediction_parameters" in param_config:
            param_config['historical_prediction_parameters']['save_path'] = 'historical_predictions/' + str(job_uuid) + '.pkl'


        self.notify_watcher(param_config=param_config,
                            control_topic='control_topic')

        fts = File(path=file_path, chunk_size=chunk_size)

        data_topic = 'data_ingestion_' + param_config['activity_title']
        create_topics(kafka_address=self.kafka_address,
                      prod_id=self.prod_id, topics=[data_topic], broker_offset=1)

        self.send_file(file_to_send=fts, topic=data_topic)

    def notify_watcher(self, param_config: dict, control_topic) -> str:

        conf = {
            "bootstrap.servers": self.kafka_address,
            "client.id": str(self.prod_id),
            "acks": 1,
            "retries": 3,
            "batch.size": 1,
            "max.in.flight.requests.per.connection": 1
        }

        producer = Producer(conf)

        headers = {'prod_id': str(self.prod_id),
                   'type': str(MessageType.control_message.value)}

        data = {'param_config': param_config}

        producer.produce(topic=control_topic,
                         value=json.dumps(data).encode('utf-8'),
                         headers=headers)
        producer.flush()

    def send_file(self, file_to_send: File, topic: str):

        chunks = read_in_chunks(file_to_read=file_to_send, CHUNK_SIZE= file_to_send.get_chunk_size())        
        conf = {
            "bootstrap.servers": self.kafka_address,
            "client.id": str(self.prod_id),
            "acks": 1,
            "retries": 3,
            "batch.size": 1,
            "max.in.flight.requests.per.connection": 1
        }
        producer = Producer(conf)

        send_data(topic = topic, chunks = chunks, 
                  file_name=file_to_send.get_name(), 
                  prod_id=str(self.prod_id), producer = producer)
