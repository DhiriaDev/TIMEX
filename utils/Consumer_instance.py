from confluent_kafka import Consumer, KafkaError, KafkaException
import os

def process_msg(msg):
    chunk = {}
    for el in msg.headers():
        chunk[el[0]] = el[1].decode()

    chunk['data'] = msg.value()

    return chunk

def consumer_receive(cons_id: int, topic : str, kafka_address :str, write_on_disk:bool):

    conf = {
        "bootstrap.servers" : kafka_address,
        "client.id" : str(cons_id),
        "group.id" : 'cons'+ str(cons_id),
        "max.in.flight.requests.per.connection" : 1,
        "auto.offset.reset":'earliest'
    }

    consumer = Consumer(conf)
    running = True
    record_list = []

    try:
        consumer.subscribe([topic])
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
                chunk = process_msg(msg)
                record_list.append(chunk)
                if int(chunk['chunk_id']) != expected_chunk_id:
                    print(f'out of order chunk {chunk["chunk_id"]} expected: {expected_chunk_id}')
#                    raise RuntimeError('out of order chunk')
                expected_chunk_id += 1
                #output_file = output_file + chunk['data']
                if int(chunk['chunk_id']) + 1 == int(chunk['chunks_number']):
                    chunks_number = int(chunk['chunks_number'])
                    running = False
        
        if len(record_list) != chunks_number:
            print(f'lenght is different {len(record_list)}, expected: {chunks_number}')
#            raise RuntimeError('lenght diff')

    except KafkaException() as e:
        print(e)
        raise
    finally:
        # Close down consumer to commit final offsets.
        consumer.close()

    if write_on_disk:
        prod_id = record_list[0]['prod_id']
        file_info = record_list[0]['file_name'].split('.')
        file_name = file_info[0] + '_cons' + str(cons_id) + '_prod' +prod_id+'.'+file_info[1]
        file_name = os.getcwd() + '/data_received/' + file_name
        print('saving file in ' + file_name)
        with open(file_name, 'wb') as fp:            
            fp.write(b''.join([item['data'] for item in record_list]))

