import base64
import pickle
import sys
import logging

from flask import Flask, request
from flask_restful import Api, Resource
import json
from random import choice, seed
from datetime import datetime

from utils import TimeSeriesContainer

app = Flask(__name__)
api = Api(app)

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

def validate (results : list[TimeSeriesContainer]) -> list[TimeSeriesContainer]:
    '''
    For the time being, the Validation module will take a model randomly.
    Successively, real validation functions will be implemented
    '''
    seed(datetime.now())
    best_model = choice(results)
    
    return best_model


class Validator(Resource):
    def post(self):

        models_results = request.form['models_results']
        predictions = json.loads(models_results)

        timeseries_containers = [] 
        [timeseries_containers.extend(
            pickle.loads(base64.b64decode(predictions.get(prediction)))
            ) for prediction in predictions ]

        try:

            if(len(timeseries_containers) > 1):
                best_model = validate(timeseries_containers)
            else:
                best_model = timeseries_containers[0]

        except ValueError as e:
            print (e)
            return 500

        best_model = pickle.dumps(best_model)
        best_model = base64.b64encode(best_model).decode('utf-8')

        return {"best_model" : best_model}, 200


api.add_resource(Validator, '/validate')

validator_address = '127.0.0.1'
validator_port = 7000

if __name__ == '__main__':
    app.run(host=validator_address,
            port=validator_port, debug=True)