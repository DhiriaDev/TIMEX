import os
import sys
import logging

from flask import Flask, request
from flask_restful import Api, Resource
import json
import pickle
import base64

from data_prediction import create_timeseries_containers

app = Flask(__name__)
api = Api(app)

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


class Predictor(Resource):

    name = ''

    def prepare_folder_for_dataSaving(path: str):
        if(not os.path.exists(path)):
            cur = ""
            print('now i will try find the path')
            for dir in (path.split('/')[:-1]):
                cur += dir
                if(not os.path.exists(cur)):
                    print('not present')
                    os.makedirs(cur)

    def extract_data(request):
        return json.loads(request.form['param_config']), pickle.loads(base64.b64decode(request.form['dataset']))

    def send_results(timeseries_containers):
        timeseries_containers = pickle.dumps(timeseries_containers)
        timeseries_containers = base64.b64encode(timeseries_containers).decode('utf-8')
        return timeseries_containers


    def post(self):

        try:
            param_config, dataset = self.extract_data(request)
            self.prepare_folder_for_dataSaving(
                param_config['historical_prediction_parameters']['save_path'])

            param_config["model_parameters"]["models"] = self.name

            timeseries_containers = create_timeseries_containers(dataset, param_config)

            logger.info('Successfully created a timeseries_container')

            return self.send_results(timeseries_containers), 200

        except ValueError as err:
            logger.error(err)
            return 500


class Arima(Predictor):
    name='arima'

class ExponentialSmoothing(Predictor):
    name = "exponentialsmoothing"

class LSTM(Predictor):
    name = 'lstm'

class Prophet(Predictor):
    name = 'fbprophet'

class MockUp(Predictor):
    name = 'mockup'

api.add_resource(MockUp, '/mockup')
api.add_resource(Arima, '/arima')
api.add_resource(ExponentialSmoothing, '/expsmooth')
api.add_resource(LSTM, '/lstm')
api.add_resource(Prophet, '/prophet')

predictor_address='127.0.0.1'
predictor_port=3000

if __name__ == '__main__':
    app.run(host = predictor_address, port = predictor_port, debug = True)
