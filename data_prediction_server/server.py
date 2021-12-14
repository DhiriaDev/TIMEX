import os, sys
import logging

from flask import Flask, request
from flask_restful import Api, Resource
import json, pickle, base64, requests
from requests.models import Response

from data_prediction.pipeline import create_timeseries_containers

app = Flask(__name__)
api = Api(app)

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

def prepare_folder_for_dataSaving(path: str):
    if(not os.path.exists(path)):
        cur = ""
        print('now i will try find the path')
        for dir in (path.split('/')[:-1]):
            cur += dir
            if(not os.path.exists(cur)):
                print('not present')
                os.makedirs(cur)

class Predictor(Resource):
    def post(self):

        try:

            param_config = json.loads(request.form['param_config'])
            dataset = request.form['dataset']

            prepare_folder_for_dataSaving(
                param_config['historical_prediction_parameters']['save_path'])

            dataset = pickle.loads(base64.b64decode(dataset))

            timeseries_containers = create_timeseries_containers(
                dataset, param_config)

            timeseries_containers = pickle.dumps(timeseries_containers)
            timeseries_containers = base64.b64encode(timeseries_containers)

            payload = {}
            payload['timeseries_containers'] = timeseries_containers

            requests.post('http://127.0.0.1:3000/renderPrediction', data=payload)

            logger.info('Successfully created a timeseries_container')
            return 200

        except ValueError as err:
            logger.error(err)
            return 500


api.add_resource(Predictor, '/predict')


if __name__ == '__main__':
    app.run(debug=True, port=4000)
