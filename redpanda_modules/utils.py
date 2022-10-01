from confluent_kafka.admin import *
from confluent_kafka import *
from multiprocessing import *

import os, sys
from math import ceil

from .constants import *

import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger(__name__)


# ----------- TOPICS UTILITY -----------

def create_topics(topics: list, client_config:dict, broker_offset : int):

    admin_config = base_config.copy()
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
    except Exception as e:
        log.error(e)


# ---------- DATA UTILITY -------------

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


def parse_msg(msg):
    decoded_msg = {}
    # msg header is a list of the type [[header_name, header_value], ...]
    for header in msg.headers():
        decoded_msg[header[0]] = header[1].decode()

    decoded_msg["data"] = msg.value()

    return decoded_msg 
    