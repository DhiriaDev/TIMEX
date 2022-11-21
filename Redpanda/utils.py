import json

from confluent_kafka.admin import *
from confluent_kafka import *
from multiprocessing import *

import os, sys, enum
from math import ceil

# from .constants import *

import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger(__name__)

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
base_config_path = os.path.join(__location__, "client_configs.json")


class MessageType(enum.Enum):
    control_message = 0
    data_message = 1


# TODO: for the time being the chunk size is hard-coded, but
# in the future we will have to parametrize it
chunk_size = 999500


# ----------- TOPICS UTILITY -----------

def create_topics(topics: list, client_config:dict, broker_offset : int):
    with open(base_config_path, "r") as f:
        config = json.load(f)

    admin_config = config["base"].copy()
    admin_config['bootstrap.servers'] = client_config['bootstrap.servers']
    admin_config['client.id'] = client_config['client.id']
    admin_client = AdminClient(admin_config)

    broker_ids = list(admin_client.list_topics().brokers.keys())
    broker_ids.sort()

    for t in range(0, len(topics)):
        if topics[t] not in admin_client.list_topics().topics:
            try:
                nt = NewTopic(topic=topics[t], num_partitions=1, replication_factor=-1,
                              replica_assignment=[[broker_ids[((t+broker_offset) % len(broker_ids))]]])

                admin_client.create_topics([nt])

                log.info(f'waiting for a leader to be assigned')
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
                log.info(f'topic {topics[t]}  successfully created')

            except Exception as e:
                log.error(e)
                continue
        else:
            log.info(f'topic {topics[t]} already exists')


# TODO: check before using! it's missing the config.
def delete_topics(client_config : dict, topics: list):
    try:
        with open(base_config_path, "r") as f:
            config = json.load(f)

        admin_config = config["base"].copy()
        admin_config['bootstrap.servers'] = client_config['bootstrap.servers']
        admin_config['client.id'] = client_config['client.id']
        admin_client = AdminClient(admin_config)

        fs = admin_client.delete_topics(topics, operation_timeout=30)

        # Wait for operation to finish.
        for topic, f in fs.items():
            try:
                f.result()  # The result itself is None
                print("Topic {} deleted".format(topic))
            except Exception as e:
                print("Failed to delete topic {}: {}".format(topic, e))

    except Exception as e:
        log.error(e)


# ---------- DATA UTILITY -------------

class File:
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
    
    def get_chunk_size(self) -> int:
        return self.chunk_size

    def get_file_size(self) -> int :
        return self.file_size


def read_in_chunks(file_to_read : File, CHUNK_SIZE):
    """
    this utility is used to read in chunks from a file written on disk
    """
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


def parse_msg(msg):
    decoded_msg = {}
    # msg header is a list of the type [[header_name, header_value], ...]
    for header in msg.headers():
        decoded_msg[header[0]] = header[1].decode()

    decoded_msg["data"] = msg.value()

    return decoded_msg 
    