from confluent_kafka.admin import *
from confluent_kafka import Producer, Consumer

from multiprocessing import *

from redpanda_utils import *
from File import *

import json
import hashlib
import os
import sys
import time
sys.path.append(".")


class RedPandaObject(Object):
    def __init__(self, config_dict: dict):
        self.config = config_dict


class Watcher(RedPandaObject):

    def __init__(self, config_dict: dict, works_to_do: list):
        self.config = config_dict
        self.consumer = Consumer(self.config)
        self.works_to_do = works_to_do

    def listen_on_control(self, control_topic: str):

        try:
            self.consumer.subscribe([control_topic])
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

                        # ------------------------------------------
                        # ---- SPAWNING THE WORKER FOR THE JOB -----
                        # ------------------------------------------
                        conf = {
                            "bootstrap.servers": self.kafka_address,
                            "client.id": str(worker_id),
                            "group.id": 'cons' + str(worker_id),
                            "max.in.flight.requests.per.connection": 1,
                            "auto.offset.reset": 'earliest'
                        }

                        param_config = json.loads(
                            parsed_msg['data'].decode('utf-8'))['param_config']

                        # TO_DO: check if the cons_id can be substituted by the job_name
                        worker = Worker(config_dict=conf,
                                        works_to_do=self.works_to_do,
                                        param_config=param_config)

                        Process(target=worker.work).start()

                    cons_id += 1
        finally:
            self.consumer.close()


class Worker(RedPandaObject):

    def __init__(self, config_dict: dict, works_to_do: list, param_config: dict):
        self.config = config_dict
        self.consumer = Consumer(self.config)

        self.param_config = param_config
        self.to_do = works_to_do

    def work(self):
        for job in self.works_to_do:
            job(self.param_config)


class JobProducer(object):
    def __init__(self, prod_id, kafka_address):
        self.prod_id = prod_id
        self.kafka_address = kafka_address
        self.admin_client = AdminClient(
            {"bootstrap.servers": self.kafka_address,
             "client.id": 'prod' + str(self.prod_id)}
        )

    def start_job(self, param_config: dict, file_path: str, chunk_size=999500):

        if file_path is None:
            print('Please insert a valid file_path for the file to send')
            exit(1)

        broker_ids = list(self.admin_client.list_topics().brokers.keys())

        # JOB NAME
        param_config['activity_title'] = hashlib.md5(
            string=(file_path+str(self.prod_id)).encode('utf-8')).hexdigest()

        self.notify_watcher(broker_ids=broker_ids, param_config=param_config)

        fts = File(path=file_path, chunk_size=chunk_size)

        self.send_file(file_to_send=fts, topic=param_config['activity_title'])

    def notify_watcher(self, broker_ids: list, param_config: dict, control_topic) -> str:

        job_name = param_config['activity_title']

        create_topics(self.admin_client, broker_ids, [job_name], 0)

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

        batch = []
        chunk_id = 0
        chunk_size = file_to_send.get_chunk_size()
        file_size = file_to_send.get_file_size()
        chunks_number = file_to_send.get_chunks_number()
        path = file_to_send.get_path()

        with open(path, 'rb') as fp:

            for chunk_id in range(0, chunks_number):

                data = read_in_chunks(fp, chunk_size)

                if data is None:
                    raise Exception("cannot read data")

                headers = {"prod_id": str(self.prod_id),
                           "chunk_id": str(chunk_id),
                           "chunks_number": str(chunks_number),
                           "file_name": file_to_send.get_name()}

                batch.append({"headers": headers, "data": data})

        conf = {
            "bootstrap.servers": self.kafka_address,
            "client.id": str(self.prod_id),
            "acks": 1,
            "retries": 3,
            "batch.size": 1,
            "max.in.flight.requests.per.connection": 1
        }

        producer = Producer(conf)

        for i in range(0, len(batch)):
            producer.produce(topic=topic, value=(batch[i])[
                             'data'], headers=batch[i]['headers'])
        producer.flush()
        print("File successfully sent")
