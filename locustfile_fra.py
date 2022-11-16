from locust import User, task, between, events
import string
import random
import json
import time

import os

from Redpanda import JobProducer

@events.init_command_line_parser.add_listener
def _(parser):
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


class RepandaClient(User):
    wait_time = between(10.0, 20.0)
    

    def __init__(self, environment):
        super().__init__(environment)
        self.kafka_address = self.environment.parsed_options.kafka_address

        with open(self.environment.parsed_options.param_config, 'r') as f:
            self.param_config = json.load(f)

        self.file_path = self.environment.parsed_options.file_path
        self.file_size = os.path.getsize(self.file_path)

        self.job_producer = JobProducer(client_id=0, kafka_address= self.kafka_address)

    @task
    def start_job(self):

        random_uuid_param_config = self.param_config.copy()
        res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        random_uuid_param_config['activity_title'] = random_uuid_param_config['activity_title'] + str(res)

        start_time = time.time()
        self.job_producer.start_job(random_uuid_param_config, self.file_path)

        result = self.job_producer.end_job()
        print('RESPONSE RECEIVED')    

        end_time = time.time()
        elapsed_time = int((end_time - start_time) * 1000)

        request_data = dict(request_type="JOB_REQUEST",
                            name=self.job_producer.job_uuid,
                            response_time=elapsed_time,
                            response_length=self.file_size)

        self.__fire_success(**request_data)

    def __fire_success(self, **kwargs):
        events.request_success.fire(**kwargs)





