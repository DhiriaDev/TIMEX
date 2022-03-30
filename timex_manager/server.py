import sys
import logging

from flask import Flask, request
import requests
from flask_restful import Api, Resource
import json
import asyncio, aiohttp
import copy

app = Flask(__name__)
api = Api(app)

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


predictor_address = 'http://timex-predictor-service:3000/' #the last part of the address will be added afterward according to the requested model
data_ingestion_address = 'http://data-ingestion-service:4000/ingest'

'''
The timex_manager and the validation_server will be deployed to the same pod.
Containers in the same pod are accessible via “localhost”, and they use the same network namespace.
'''
validator_address = 'http://localhost:7000/validate'

available_predictors = ['arima', 'fbprophet', 'lstm', 'mockup', 'exponentialsmoothing']

def prepare_params(param_config:dict, models : list) -> dict :

    configurations_for_each_model = dict.fromkeys(models, {})
    hist_paths = []

    # parsing a google drive link
    dataset_url = param_config['input_parameters']['source_data_url']
    dataset_url = 'https://drive.google.com/uc?export=download&confirm=t&id=' + \
                    dataset_url.split('/')[-2]
    param_config['input_parameters']['source_data_url'] = dataset_url
    

    for model in models:
        model_config = copy.deepcopy(param_config)

        model_config["model_parameters"]["models"] = model

        # Let's check the historical predictions (if requested)
        if 'historical_prediction_parameters' in param_config:            
            
            if len(hist_paths) == 0 : #list __init__ just at the first iteration
                hist_paths = (param_config['historical_prediction_parameters']['save_path']).split(',')
            
            # Check if the historical predictions are requested for the current model
            hist_path = ''
            for path in hist_paths:
                if (model in path):
                    hist_path = path

            if hist_path == '':
                model_config.pop('historical_prediction_parameters')
            else :
                model_config['historical_prediction_parameters']['save_path']=hist_path
        
        configurations_for_each_model[model]=model_config
    
    logger.info("configurations for each model completed")
    return configurations_for_each_model


async def call_predictor(session : aiohttp.ClientSession, model : str, url : str, models_results : dict, payload : dict):
    async with session.post(url=url, data=payload) as resp:
        logger.info('Asking for: '+ url)       
        results = await (resp.json())
        results = results['timeseries_containers']
        models_results[model]= results


async def schedule_coroutines(models: list, param_config : dict, dataset) -> dict:
    
    configurations_for_each_model = prepare_params(param_config, models)
    session_timeout = aiohttp.ClientTimeout(total=None) #timeout disabled

    async with aiohttp.ClientSession(timeout=session_timeout) as session:
        models_results = dict.fromkeys(models, {})
        async_requests=[call_predictor(session, 
                                        model, 
                                        url = str(predictor_address+model.lower()),
                                        payload = { 
                                            "param_config" : json.dumps(configurations_for_each_model[model]),
                                            "dataset" : dataset
                                            },
                                        models_results=models_results) for model in models]
        await asyncio.gather(*async_requests)
        return models_results


class Manager(Resource):

    def post(self):
        
        logger.info('Request received')
        param_config = json.loads(request.form['param_config'])

        # First, we check if the requested model(s) is(are) valid.
        # If not, the request is not even forwarded to the data ingestion module
        models = [*param_config["model_parameters"]["models"].split(",")]
        if not all(item in available_predictors for item in models):
            logger.error('One or more of the requested models is not valid')
            return 400


        try:
            logger.info('Contacting the data ingestion module')
            # here data has been sent to the data ingestion module
            ingestion_resp = json.loads(requests.post(
                data_ingestion_address, 
                data=request.form #we can directly forward the request from the web_app
                ).text)

            logger.info('Data received')

        except ValueError as e:
            print(e)
            return 'Error in contacting the data ingestion module'

            
        try:
            logger.info('Contacting the Predictor!')
            results = asyncio.run(schedule_coroutines(models, param_config, ingestion_resp['dataset']))
            logger.info('Prediction results for each requested model received.')

        except ValueError as e:
            print(e)
            return 'Error in contacting the predictor'
        
        try:
            logger.info('Contacting the validation server to get the best model')
            best_model = json.loads(requests.post(
                validator_address,
                data = {'param_config' : json.dumps(param_config), 
                        'models_results' : json.dumps(results)}
            ).text)
        except ValueError as e:
            print(e)
            return 'Error in contacting the validation server'

        #Returning the prediction of the model with the best results
        return best_model, 200 


api.add_resource(Manager, '/predict')

address = '127.0.0.1'
docker_address= '0.0.0.0'
port = 6000

if __name__ == '__main__':
    app.run(host=docker_address,
            port=port, debug=True)
