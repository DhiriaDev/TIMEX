from confluent_kafka.admin import *
import enum

class MessageType(enum.Enum):
    control_message = 0

def parse_msg(msg):
    decoded_msg = {}
    # msg header is a list of the type [[header_name, header_value], ...]
    for header in msg.headers():
        decoded_msg[header[0]] = header[1].decode()

    decoded_msg["data"] = msg.value()

    return decoded_msg


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