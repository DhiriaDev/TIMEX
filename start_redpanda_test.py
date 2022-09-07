import os, sys, argparse
sys.path.append(".")

from Utils import *


from confluent_kafka.admin import *
from confluent_kafka import Producer

import hashlib
import json

kafka_address = '0.0.0.0:9092'
control_topic = 'control_topic'
admin_client = AdminClient(
                        {"bootstrap.servers" : kafka_address,
                         "client.id" : 'test'}
                        )

prod_id = 0
chunk_size = 999500

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


def prod_split(file_to_send : File, topic : str, chunk_size : int, kafka_address : str, prod_id : int): 
 
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

    
def read_in_chunks(file_object, CHUNK_SIZE):
    data = file_object.read(CHUNK_SIZE)
    if not data:
        return None
    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", 
                        help = "path to file", type=str)

    path = parser.parse_args().path

    if path is None:
        print('Please insert a valid path for the file to send')
        exit(1)

    broker_ids = list(admin_client.list_topics().brokers.keys())

    job_name = ping_watcher(path, broker_ids)

    fts = File(path, chunk_size=chunk_size)

    prod_split(fts, job_name, chunk_size, kafka_address, prod_id)
    