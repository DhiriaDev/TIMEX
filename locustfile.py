from locust import User, task, between, events
import string
import random
from modules.job_producer import produce_test
import json

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

class LocustUser(User):
    wait_time = between(30, 40)

    @task
    def make_test(self):
        res = ''.join(random.choices(string.ascii_uppercase +
                                     string.digits, k=10))
        json_file = open(self.environment.parsed_options.param_config, 'r')
        param_config = json.load(json_file)
        json_file.close()
        param_config["activity_title"] = param_config["activity_title"] + str(res)
        result = produce_test(self.environment.parsed_options.kafka_address, param_config, self.environment.parsed_options.file_path)
        print("test terminated")