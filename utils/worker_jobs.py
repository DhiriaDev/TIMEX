from confluent_kafka.admin import *
from redpanda_utils import *

def receive_data(new_topic: str):
    running = True
    record_list = None

    try:
        consumer.subscribe([new_topic])
        expected_chunk_id = 0

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
