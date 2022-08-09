import sys, os
from confluent_kafka import Producer

sys.path.append(".") # needed to import parent folders
from Utils import File

def read_in_chunks(file_object, CHUNK_SIZE):
    data = file_object.read(CHUNK_SIZE)
    if not data:
        return None
    return data


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