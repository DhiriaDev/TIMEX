import sys
import logging

sys.path.append('./')
log = logging.getLogger(__name__)

import argparse
import json

from Redpanda import JobProducer


def produce_test(kafka_address, param_config, file_path):
    job_producer = JobProducer(client_id=0, kafka_address=kafka_address)
    job_producer.start_job(param_config, file_path)
    results = job_producer.end_job()
    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'kafka_address',
        type=str,
        help='single address (or list of addresses) of the form IP:port[,IP:port]'
    )

    parser.add_argument(
        'param_config',
        type=str,
        help='Path to the json file from which to load the param config of the job'
    )

    parser.add_argument(
        'file_path',
        type=str,
        help='Path to were take the input file that is to be sent for the job. The absolute path is suggested'
    )

    args = parser.parse_args()
    kafka_address = args.kafka_address
    param_config_path = args.param_config
    file_path = args.file_path

    if kafka_address is None:
        log.error('a kafka address has been not specified')
        exit(1)

    if param_config_path is None:
        log.error('a path to a param_config has been not specified')
        exit(1)

    if file_path is None:
        log.error('a file path has been not specified')
        exit(1)

    # Opening JSON file
    json_file = open(param_config_path, 'r')
    param_config = json.load(json_file)
    json_file.close()
    produce_test(kafka_address, param_config, file_path)
