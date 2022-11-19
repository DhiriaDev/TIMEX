from locust import TaskSet, task, User, between, Locust, events
import string
import random
from modules.job_producer import produce_test
import json
import time

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


class CustomClient:
    def __init__(self, environment):
        self.environment = environment

    def custom_req(self):
        try:
            start_time=time.time()

            res = ''.join(random.choices(string.ascii_uppercase +
                          string.digits, k=10))
            json_file = open(self.environment.parsed_options.param_config, 'r')
            param_config = json.load(json_file)
            json_file.close()
            param_config["activity_title"] = param_config["activity_title"] + str(res)
            result = produce_test(self.environment.parsed_options.kafka_address, param_config, self.environment.parsed_options.file_path)
            total_time = int ((time.time() - start_time)*1000)

            print("success, time taken", total_time)
            self.environment.events.request_success.fire(
                request_type="WSR", name="prova", response_length=1, response_time=total_time
            )

        except Exception as e:
            total_time = int (time.time() - start_time)*1000
            #self.environment.events.request_success.fire(
            #    request_type="WSR", name="prova", response_length=1, response_time=total_time
            #)
            print("failure, time taken", total_time)


class CustomLocust(User):
    abstract = True
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.client = CustomClient(self.environment)


class UserBehaviour(TaskSet):
    @task()
    def my_task(self):
        self.client.custom_req()


class MyUser(CustomLocust):
    tasks = [UserBehaviour]
    wait_time = between(0.0, 0.1)
    host = "example.com"
