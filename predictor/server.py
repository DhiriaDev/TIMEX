import os
import sys
import logging

from flask import Flask, request
from flask_restful import Api, Resource
import json
import pickle
import base64
from joblib import parallel_backend

from pytest import param

from data_prediction import create_timeseries_containers

app = Flask(__name__)
api = Api(app)



logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


def prepare_folder_for_dataSaving(path: str):
    if(not os.path.exists(path)): # If historical predictions non present yet
        cur = ""
        print('Finding the path...')
        for dir in (path.split('/')[:-1]):
            cur += dir
            if(not os.path.exists(cur)):
                print('not present')
                os.makedirs(cur)


def predict(request: request):
    try:
        param_config = json.loads(request.form['param_config'])
        dataset = pickle.loads(base64.b64decode(request.form['dataset']))
        logger.info(
            "Configuration parameters and dataset successfully received")

        if 'historical_prediction_parameters' in param_config:
            prepare_folder_for_dataSaving(
                param_config['historical_prediction_parameters']['save_path'])


        timeseries_containers = create_timeseries_containers(
            dataset, param_config)

        logger.info('Successfully created a timeseries_container')

        timeseries_containers = pickle.dumps(timeseries_containers)
        timeseries_containers = base64.b64encode(timeseries_containers).decode('utf-8')

        payload = {}
        payload['timeseries_containers'] = timeseries_containers

        return payload, 200

    except ValueError as err:
        logger.error(err)
        return 500

class Arima(Resource):
    def post(self):
        logger.info('Arima: post request received')
        return predict(request)
class ExponentialSmoothing(Resource):
    def post(self):
        logger.info('ExponentialSmoothing: post request received')
        return predict(request)
class LSTM(Resource):
    def post(self):
        logger.info('LSTM: post request received')
        return predict(request)
class Prophet(Resource):
    def post(self):
        logger.info('Prophet: post request received')
        return predict(request)
class MockUp(Resource):
    def post(self):
        logger.info('MockUp: post request received')
        return predict(request)


api.add_resource(Arima, '/arima')
api.add_resource(MockUp, '/mockup')
api.add_resource(ExponentialSmoothing, '/exponentialsmoothing')
api.add_resource(LSTM, '/lstm')
api.add_resource(Prophet, '/fbprophet')


predictor_address = '127.0.0.1'
predictor_port = 3000

if __name__ == '__main__':
    app.run(host=predictor_address, port=predictor_port, debug=True)
