import os, sys, time
sys.path.append(".")

import enum
import hashlib
import json

from confluent_kafka.admin import *
from redpanda_utils import *
from File import *

def parse_msg(msg):
    decoded_msg = {}
    # msg header is a list of the type [[header_name, header_value], ...]
    for header in msg.headers():
        decoded_msg[header[0]] = header[1].decode()

    decoded_msg["data"] = msg.value()

    return decoded_msg




class RedPandaObject(Object):    
    def __init__(self, config_dict : dict):
        self.config = config_dict


class Watcher(RedPandaObject):

    def __init__(self, config_dict : dict, works_to_do : list):
        self.config = config_dict
        self.consumer = Consumer(self.config)
        self.to_do = works_to_do

    def listen_on_control (control_topic : str):

        try:
            consumer.subscribe([control_topic])
            running = True
            worker_id = 0

            while running:
                msg = consumer.poll(timeout=1)
                if msg is None: continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # End of partition event
                        sys.stderr.write('%% %s [%d] reached end at offset %d\n' %
                                        (msg.topic(), msg.partition(), msg.offset()))
                    elif msg.error():
                        raise KafkaException(msg.error())
                else:
                    parsed_msg = parse_msg(msg)
                    msg_type =  MessageType(int(parsed_msg['type'])).name

                    if msg_type == 'control_message':
                        
                        # ------------------------------------------
                        # ---- SPAWNING THE WORKER FOR THE JOB -----
                        # ------------------------------------------
                        conf = {
                            "bootstrap.servers" : kafka_address,
                            "client.id" : str(worker_id),
                            "group.id" : 'cons'+ str(worker_id),
                            "max.in.flight.requests.per.connection" : 1,
                            "auto.offset.reset":'earliest'
                        }

                        # TO_DO: check if the cons_id can be substituted by the job_name
                        worker = Worker(config_dict = conf, works_to_do = works_to_do)                        
                        
                        param_config = json.loads(parsed_msg['data'].decode('utf-8'))

                        Process(target= worker.work(param_config = param_config)).start()

                    cons_id += 1
        finally:
            consumer.close()


class Worker(RedPandaObject):

    def __init__(self, config_dict : dict, works_to_do : list):
        self.config = config_dict
        self.consumer = Consumer(self.config)
        self.to_do = works_to_do


    def work(param_config : dict):
        self.param_config = param_config
        
        for job in self.works_to_do:
            job(param_config)


    def receive_data(new_topic : str):
        running = True
        record_list = None

        try:
            consumer.subscribe([new_topic])
            expected_chunk_id = 0

            while running:
                msg = consumer.poll(timeout=1.0)
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
                    decoded_msg = parse_msg(msg)
                    chunk_id = int(decoded_msg["chunk_id"])
                    chunks_number = int(decoded_msg["chunks_number"])

                    if chunk_id + 1 > chunks_number:
                        sys.stderr.write("Unexpected msg chunk id")

                    if record_list is None:
                        record_list = [{} for i in range(chunks_number)]

                    record_list[chunk_id] = decoded_msg

                    # el is a dictionary and to check if it is empty it suffices to check if its len == 0
                    still_to_receive_chunks = True in (len(el) == 0 for el in record_list)

                    if not still_to_receive_chunks:
                        running = False

            if len(record_list) != chunks_number:
                raise RuntimeError(
                    f"lenght is different {len(record_list)}, expected: {chunks_number}"
                )

        except KafkaException() as e:
            print(e)
            raise
        finally:
            # Close down consumer to commit final offsets.
            consumer.close()

        if write_on_disk:
            prod_id = record_list[0]["prod_id"]
            file_info = record_list[0]["file_name"].split(".")
            file_name = (
                file_info[0]
                + "_cons"
                + str(cons_id)
                + "_prod"
                + prod_id
                + "."
                + file_info[1]
            )
            file_name = os.getcwd() + "/data_received/" + file_name
            print("saving file in " + file_name)
            with open(file_name, "wb") as fp:
                fp.write(b"".join([item["data"] for item in record_list]))

class Producer(object):
    def __init__(self, prod_id, kafka_address, control_topic):
        self.prod_id = 0
        self.kafka_address = kafka_address
        self.control_topic = control_topic
        admin_client = AdminClient(
                                {"bootstrap.servers" : self.kafka_address,
                                "client.id" : 'test'}
                                )




    def ping_watcher(path : str, broker_ids : list) -> str:
        job_name = hashlib.md5(string= (path+str(prod_id)).encode('utf-8')).hexdigest()
        #create_topics(admin_client, broker_ids, [control_topic], 0)
        create_topics(admin_client, broker_ids, [job_name], 0)

        conf = {
            "bootstrap.servers" : kafka_address, 
            "client.id": str(prod_id), 
            "acks":1,
            "retries":3,
            "batch.size":1,
            "max.in.flight.requests.per.connection":1
        }

        producer = Producer(conf)


        headers = {'prod_id' : str(prod_id),
                'type' : str(MessageType.control_message.value)}

        data = {'topic' : job_name}


        producer.produce(topic = control_topic,
                            value = json.dumps(data).encode('utf-8'),
                            headers = headers)

        producer.flush()

        return job_name

    def send_file(file_to_send : File, topic : str, chunk_size : int, kafka_address : str, prod_id : int): 
    
        batch = []
        chunk_id = 0
        file_size = file_to_send.get_file_size()
        chunks_number = file_to_send.get_chunks_number()
        path = file_to_send.get_path()

        
        with open(path, 'rb') as fp:

            for chunk_id in range(0, chunks_number):   

                
                data = read_in_chunks(fp, chunk_size)

                if data is None:
                    raise Exception("cannot read data")

                headers = {"prod_id": str(prod_id), 
                            "chunk_id": str(chunk_id),
                            "chunks_number": str(chunks_number),
                            "file_name" : file_to_send.get_name()}

                batch.append({"headers" : headers , "data" : data})

        conf = {
            "bootstrap.servers" : kafka_address, 
            "client.id": str(prod_id), 
            "acks":1,
            "retries":3,
            "batch.size":1,
            "max.in.flight.requests.per.connection":1
        }

        producer = Producer(conf)

        for i in range(0,len(batch)):
            producer.produce(topic = topic, value = (batch[i])['data'], headers = batch[i]['headers'])

        producer.flush()
        print("File successfully sent")
        return
