from confluent_kafka import Consumer

c = Consumer({
    'bootstrap.servers': 'zk1.dhiria.com,zk2.dhiria.com,zk4.dhiria.com',
    'group.id': 'mygroup',
    'auto.offset.reset': 'earliest',
    'security.protocol' : 'sasl_ssl',
    # CA certificate file for verifying the broker's certificate.
    'ssl.ca.location' : './redpanda-ca.crt',
    'sasl.username' : 'dhiria',
    'sasl.password' : 'piic9xplo8fc',
    'sasl.mechanisms' : 'SCRAM-SHA-256'
})

c.subscribe(['control_topic'])

while True:
    msg = c.poll(1.0)

    if msg is None:
        continue
    if msg.error():
        print("Consumer error: {}".format(msg.error()))
        continue

    print('Received message: {}'.format(msg.value().decode('utf-8')))

c.close()