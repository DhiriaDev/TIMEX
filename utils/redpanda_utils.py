from confluent_kafka.admin import *
import sys
import enum

# ----------- TOPICS UTILITY -----------
def create_topics(admin_client: AdminClient, broker_ids : list, topics : list, broker_offset):
    topics_list = []
    broker_ids.sort()
    print(broker_ids)
    for t in range(0, len(topics)):
        nt = NewTopic(topic =topics[t], num_partitions = 1, replication_factor = -1, 
                        replica_assignment = [[broker_ids[((t+broker_offset) % len(broker_ids))]]])
        #nt = NewTopic(topic = topics[t], num_partitions = 1, replication_factor = 1)
        #nt = NewTopic(name=topics[t], num_partitions = 3, replication_factor = 1)
        topics_list.append(nt)
        futures = admin_client.create_topics([nt])
        #list(futures.values())[0].result()
        print('topic ' + topics[t] + ' successfully created')
        found_leader = False
        while not(found_leader):
            try:
                leader = admin_client.list_topics().topics[topics[t]].partitions[0].leader
                if leader != -1:
                    found_leader = True
                    print('leader found: %s' %(leader))
                else:
                    print('leader not found')
            except:
                continue
        

def delete_topics(admin_client: AdminClient, topics : list):
    try:
        admin_client.delete_topics(topics=topics)
        print("Topics Deleted Successfully")
    except UnknownTopicOrPartitionError as e:
        print("A Topic Doesn't Exist")
    except  Exception as e:
        print(e)

#---------- MESSAGES UTILITY -------------

def read_in_chunks(file_object, CHUNK_SIZE):
    data = file_object.read(CHUNK_SIZE)
    if not data:
        return None
    return data


def receive_data(new_topic: str, consumer, write_on_disk = False):
    running = True
    record_list = None

    try:
        consumer.subscribe([new_topic])

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

class MessageType(enum.Enum):
    control_message = 0

def parse_msg(msg):
    decoded_msg = {}
    # msg header is a list of the type [[header_name, header_value], ...]
    for header in msg.headers():
        decoded_msg[header[0]] = header[1].decode()

    decoded_msg["data"] = msg.value()

    return decoded_msg
