
'''
max.in.flight.requests.per.connection:
    Maximum number of in-flight requests per broker connection. 
    This is a generic property applied to all broker communication, however it is primarily relevant to produce requests. 
    In particular, note that other mechanisms limit the number of outstanding consumer fetch request per broker to one.
    Type: integer
'''
base_config = {
    'bootstrap.servers' : "",
    "client.id" : "",
    "max.in.flight.requests.per.connection": 5, #(this is the max we can afford. See enable.idempotence in producer config)
    'receive.message.max.bytes' : 2000000000,
    'security.protocol' : 'sasl_ssl',
    # CA certificate file for verifying the broker's certificate.
    'ssl.ca.location' : './redpanda-ca.crt',
    'sasl.username' : 'dhiria',
    'sasl.password' : 'piic9xplo8fc',
    'sasl.mechanisms' : 'SCRAM-SHA-256'
} 

#########################################################################

'''
session.timeout.ms :
    Client group session and failure detection timeout. The consumer sends periodic heartbeats (heartbeat.interval.ms)
    to indicate its liveness to the broker. If no hearts are received by the broker for a group member within
    the session timeout, the broker will remove the consumer from the group and trigger a rebalance.
    The allowed range is configured with the broker configuration properties 
    group.min.session.timeout.ms and group.max.session.timeout.ms. Also see max.poll.interval.ms.
    Type: integer

max.poll.interval.ms :
    Maximum allowed time between calls to consume messages (e.g., rd_kafka_consumer_poll()) for high-level consumers.
    If this interval is exceeded the consumer is considered failed and the group will rebalance in order to reassign 
    the partitions to another consumer group member. Warning: Offset commits may be not possible at this point. 
    Note: It is recommended to set enable.auto.offset.store=false for long-time processing applications and
    then explicitly store offsets (using offsets_store()) after message processing, to make sure offsets are 
    not auto-committed prior to processing has finished. The interval is checked two times per second. 

enable.auto.offset.store :
    Automatically store offset of last message provided to application. The offset store is an in-memory store 
    of the next offset to (auto-)commit for each partition.
    Type: boolean

enable.auto.commit :
    Automatically and periodically commit offsets in the background. Note: setting this to false does not prevent 
    the consumer from fetching previously committed start offsets. To circumvent this behaviour set specific start 
    offsets per partition in the call to assign().
    Type: boolean

auto.offset.reset:
    Action to take when there is no initial offset in offset store or the desired offset 
    is out of range: 'smallest','earliest' - automatically reset the offset to the smallest offset, 
    'largest','latest' - automatically reset the offset to the largest offset, 'error' - 
    trigger an error (ERR__AUTO_OFFSET_RESET) which is retrieved by consuming messages and checking 'message->err'.
    Type: enum value
'''

default_consumer_config = base_config.copy()
default_consumer_config.update(
    {
        "group.id" : "",
        "session.timeout.ms" : 30000,
        "max.poll.interval.ms" : 30000,
        "enable.auto.offset.store" : False,
        "enable.auto.commit" : True,
        # "enable.partition.eof" : True, #TODO check if it is a valid option
        # "allow.auto.create.topics" : True, #TODO check if it is a valid option
        "auto.offset.reset": 'earliest',
        'fetch.message.max.bytes' : 1000000000,
    }
) 

#########################################################################

'''
enable.idempotence:
    When set to true, the producer will ensure that messages are successfully produced exactly once 
    and in the original produce order. The following configuration properties are adjusted automatically 
    (if not modified by the user) when idempotence is enabled: 
        max.in.flight.requests.per.connection=5 (must be less than or equal to 5), 
        retries=INT32_MAX (must be greater than 0), 
        request.required.acks=all, 
        queuing.strategy=fifo. 
    Producer instantation will fail if user-supplied configuration is incompatible.
    Type: boolean

batch.num.messages:
    Maximum number of messages batched in one MessageSet. The total MessageSet 
    size is also limited by batch.size and message.max.bytes.
    Type: integer

batch.size:
    Maximum size (in bytes) of all messages batched in one MessageSet, including protocol framing overhead. 
    This limit is applied after the first message has been added to the batch, regardless 
    of the first message's size, this is to ensure that messages that exceed batch.size are produced. 
    The total MessageSet size is also limited by batch.num.messages and message.max.bytes.
    Type: integer

request.required.acks:
    This field indicates the number of acknowledgements the leader broker must receive
    from ISR (In-Sync Replica set) brokers before responding to the request: 0=Broker does not send 
    any response/ack to client, -1 or all=Broker will block until message is committed by all in sync replicas (ISRs). 
    If there are less than min.insync.replicas (broker configuration) in the ISR set the produce request will fail.
    Type: integer
'''

default_producer_config = base_config.copy()
default_producer_config.update(
    {
        "enable.idempotence" : True,
        # "queue.buffering.backpressure.threshold" : int , #TODO check if it is a valid option
        # "compression.codec" : enum , #TODO check if it is a valid option
        "batch.num.messages" : 10000, # (== default value)
        "batch.size" : 1000000 #(== default value)
    }
) 