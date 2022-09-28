from confluent_kafka.admin import *
from confluent_kafka import *

import os
import sys
import enum
import time
from math import ceil

import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger(__name__)


# ----------- TOPICS UTILITY -----------


def create_topics(topics: list, client_config:dict, broker_offset : int):

    admin_client = AdminClient(
            {   
                "bootstrap.servers": client_config['bootstrap.servers'],
                "client.id": client_config['client.id']
            }
        )

    broker_ids = list(admin_client.list_topics().brokers.keys())
    broker_ids.sort()

    for t in range(0, len(topics)):
        if topics[t] not in admin_client.list_topics().topics:
            try:
                nt = NewTopic(topic=topics[t], num_partitions=1, replication_factor=-1,
                                replica_assignment=[[broker_ids[((t+broker_offset) % len(broker_ids))]]])

                admin_client.create_topics([nt])
                log.info(f'topic {topics[t]}  successfully created')

                log.debug(f'waiting for a leader to be assigned')
                found_leader = False
                while not (found_leader):
                    try:
                        leader = admin_client.list_topics(
                        ).topics[topics[t]].partitions[0].leader
                        if leader != -1:
                            found_leader = True
                            log.debug(f'leader found: {leader}')
                        else:
                            log.debug(f'leader not found')
                    except KeyError:
                        continue

            except Exception as e:
                log.error(e)
                continue
        else:
            log.info(f'topic {topics[t]} already exists')


def delete_topics(client_config : dict, topics: list):
    try:
        admin_client = AdminClient(
            {   
                "bootstrap.servers": client_config['bootstrap.servers'],
                "client.id": client_config['client.id']
            }
        )
        admin_client.delete_topics(topics=topics)
        print("Topics Deleted Successfully")
    except UnknownTopicOrPartition as e:
        print("A Topic Doesn't Exist")
    except Exception as e:
        print(e)


# ---------- DATA UTILITY -------------


def receive_data(topic: str, consumer_config : dict):
    running = True
    record_list = None
    
    admin_client = AdminClient(
        {   
            "bootstrap.servers": consumer_config['bootstrap.servers'],
            "client.id": consumer_config['client.id']
        }
    )

    consumer = Consumer(consumer_config)
    try:
        while topic not in admin_client.list_topics().topics :
            log.debug(f'the topic {topic} does not exist: waiting..')
            time.sleep(2)

        consumer.subscribe([topic])

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
                still_to_receive_chunks = True in (
                    len(el) == 0 for el in record_list)

                if not still_to_receive_chunks:
                    running = False

        if len(record_list) != chunks_number:
            raise RuntimeError(
                f"length is different {len(record_list)}, expected: {chunks_number}"
            )

    except KafkaException() as e:
        log.error(e) 
    finally:
        # Close down consumer to commit final offsets.
        consumer.close()

    data = b"".join([item["data"] for item in record_list])
    return data


def send_data(topic: str, chunks: list, file_name :str, producer_config : dict):
    chunks_number = len(chunks)
    producer = Producer(producer_config)
    for i in range(0,chunks_number):
        msg_header = {
            "prod_id": producer_config['client.id'],
            "chunk_id": str(i),
            "chunks_number": str(chunks_number),
            "file_name": file_name}

        producer.produce(topic=topic, value=chunks[i], headers=msg_header)

    producer.flush()
    print("File successfully sent")


class File():
    def __init__(self, path : str, chunk_size: int):
        self.path = path
        self.name = path.split('/')[-1]
        self.file_size = os.path.getsize(path)

        self.chunk_size = chunk_size
        self.chunks_number = ceil(float(self.file_size) / float(self.chunk_size))

    def get_chunks_number(self) -> int :
        return self.chunks_number

    def get_name (self) -> str:
        return self.name
 
    def get_path(self) -> str :
        return self.path
    
    def get_chunks_number(self) -> int:
        return self.chunks_number
    
    def get_chunk_size(self) -> int:
        return self.chunk_size

    def get_file_size(self) -> int :
        return self.file_size


def read_in_chunks(file_to_read : File, CHUNK_SIZE):
    '''
    this utility is used to read in chunks from a file written on disk
    '''
    chunks = []
    chunks_number = file_to_read.get_chunks_number()
    path = file_to_read.get_path()

    with open(path, 'rb') as fp:
            for _ in range(0, chunks_number):
                data = fp.read(CHUNK_SIZE)
                if data is None:
                    raise Exception("Cannot read data")
                else:
                    chunks.append(data)

    return chunks


def prepare_chunks(data, CHUNK_SIZE):
    '''
    This utility is used to divide a given data structure (base64Encoded) in chunks 
    '''
    data_size = len(data)
    chunks_number =ceil(float(data_size) / float(CHUNK_SIZE))
    chunks = []
    for i in range(0, chunks_number):
        chunks.append(data[i * CHUNK_SIZE : min(data_size, (i+1) *CHUNK_SIZE)])
    
    assert(len(chunks) != 0)
    return chunks


# ---------- MESSAGES UTILITY -------------
class MessageType(enum.Enum):
    control_message = 0

def parse_msg(msg):
    decoded_msg = {}
    # msg header is a list of the type [[header_name, header_value], ...]
    for header in msg.headers():
        decoded_msg[header[0]] = header[1].decode()

    decoded_msg["data"] = msg.value()

    return decoded_msg