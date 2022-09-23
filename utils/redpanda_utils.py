import time
from confluent_kafka.admin import *
from confluent_kafka import *

import sys
import enum
from math import ceil
from .File import *

import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger(__name__)


# ----------- TOPICS UTILITY -----------


def create_topics(kafka_address, prod_id, topics: list, broker_offset):
    topics_list = []

    admin_client = AdminClient(
            {"bootstrap.servers": kafka_address,
             "client.id": 'prod' + str(prod_id)}
        )

    broker_ids = list(admin_client.list_topics().brokers.keys())
    broker_ids.sort()

    for t in range(0, len(topics)):
        if topics[t] not in admin_client.list_topics().topics:
            nt = NewTopic(topic=topics[t], num_partitions=1, replication_factor=-1,
                            replica_assignment=[[broker_ids[((t+broker_offset) % len(broker_ids))]]])
            topics_list.append(nt)
            admin_client.create_topics([nt])
            print('topic ' + topics[t] + ' successfully created')
            found_leader = False
            while not (found_leader):
                try:
                    leader = admin_client.list_topics(
                    ).topics[topics[t]].partitions[0].leader
                    if leader != -1:
                        found_leader = True
                        print('leader found: %s' % (leader))
                    else:
                        print('leader not found')
                except KeyError:
                    continue
        else:
            print('topic', topics[t], 'already exists')


def delete_topics(admin_client: AdminClient, topics: list):
    try:
        admin_client.delete_topics(topics=topics)
        print("Topics Deleted Successfully")
    except UnknownTopicOrPartitionError as e:
        print("A Topic Doesn't Exist")
    except Exception as e:
        print(e)


# ---------- DATA UTILITY -------------

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


def receive_data(topic: str, consumer, kafka_address, cons_id):
    running = True
    record_list = None

    admin_client = AdminClient(
        {"bootstrap.servers": kafka_address,
            "client.id": str(cons_id)}
    )
    try:
        while topic not in admin_client.list_topics().topics :
            log.info('the topic', topic, 'does not exist: waiting..')
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


def send_data(topic: str, chunks: list, file_name :str, prod_id , producer: Producer):
    chunks_number = len(chunks)
    for i in range(0,chunks_number):
        msg_header = {"prod_id": str(prod_id),
                    "chunk_id": str(i),
                    "chunks_number": str(chunks_number),
                    "file_name": file_name}

        producer.produce(topic=topic, value=chunks[i], headers=msg_header)

    producer.flush()
    print("File successfully sent")


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
