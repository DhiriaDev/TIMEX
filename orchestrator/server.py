import sys
import logging

from flask import Flask, request
from flask_restful import Api, Resource
import json
import requests


app = Flask(__name__)
api = Api(app)

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


predictor_address = 'http://127.0.0.1:3000/'
available_predictors = ['arima', 'fbprophet', 'lstm', 'mockup', 'exponentialsmoothing']

class Orchestrator(Resource):

    def post(self):
        
        logger.info('Request received')

        param_config = json.loads(request.form['param_config'])

        models = [*param_config["model_parameters"]["models"].split(",")]

        # Let's check if the requested model(s) is(are) valid
        if not all(item in available_predictors for item in models):
            logger.error('One or more of the requested models is not valid')
            return 400

        
        try:            
            
            models_result = dict.fromkeys(models, {}) 

            for model in models:
                #address = predictor_address + model.lower()
                address=str(predictor_address+model.lower())
                logger.info('asking for: '+address)
                results = json.loads(
                        requests.post(
                            url=address, data=request.form).text
                        )['timeseries_containers']
                
                models_result[model]=results

        except ValueError as err:
            logger.error(err)
            return 500

        '''
        Future functionalities will be implemented, such as:
         - calling a validation service to extract the best predictor
        --------------------------------------------------------------------------------------
        '''

        payload = {"models_results" : models_result}

        return payload, 200


api.add_resource(Orchestrator, '/predict')

orchestrator_address = '127.0.0.1'
orchestrator_port = 6000

if __name__ == '__main__':
    app.run(host=orchestrator_address,
            port=orchestrator_port, debug=True)
