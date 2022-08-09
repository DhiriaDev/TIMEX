import os, time
import enum

from confluent_kafka.admin import *


class MessageType(enum.Enum):
    control_message = 0

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




def parse_msg(msg):
    chunk = {}
    for el in msg.headers():
        chunk[el[0]] = el[1].decode()

    chunk['data'] = msg.value()
    return chunk




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
        record_list = []

        try:
            consumer.subscribe([new_topic])
            expected_chunk_id = 0 

            while running:
                msg = consumer.poll(timeout=1.0)
                if msg is None: continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # End of partition event
                        sys.stderr.write('%% %s [%d] reached end at offset %d\n' %
                                        (msg.topic(), msg.partition(), msg.offset()))
                    elif msg.error():
                        print(msg.error())
                        raise KafkaException(msg.error())
                else:
                    chunk = parse_msg(msg)
                    record_list.append(chunk)

                    if int(chunk['chunk_id']) != expected_chunk_id:
                        print(f'out of order chunk {chunk["chunk_id"]} expected: {expected_chunk_id}')

                    expected_chunk_id += 1

                    if int(chunk['chunk_id']) + 1 == int(chunk['chunks_number']):
                        chunks_number = int(chunk['chunks_number'])
                        running = False
            
            if len(record_list) != chunks_number:
                print(f'length is different {len(record_list)}, expected: {chunks_number}')

        except KafkaException() as e:
            print(e)
            raise
        finally:
            # Close down consumer to commit final offsets.
            consumer.close()

        msg_payload = b''.join([item['data'] for item in record_list])


        if write_on_disk:
            prod_id = record_list[0]['prod_id']
            file_info = record_list[0]['file_name'].split('.')
            file_name = file_info[0] + '_cons' + str(cons_id) + '_prod' +prod_id+'.'+file_info[1]
            file_name = os.getcwd() + '/data_received/' + file_name
            print('saving file in ' + file_name)
            with open(file_name, 'wb') as fp:            
                fp.write(b''.join([item['data'] for item in record_list]))
