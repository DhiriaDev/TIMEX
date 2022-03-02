import sys
import logging

from flask import Flask, request
import requests
from flask_restful import Api, Resource
import json
import asyncio, aiohttp

app = Flask(__name__)
api = Api(app)

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


predictor_address = 'http://127.0.0.1:3000/'
data_ingestion_address = 'http://127.0.0.1:4000/ingest'

available_predictors = ['arima', 'fbprophet', 'lstm', 'mockup', 'exponentialsmoothing']

async def call_predictor(session : aiohttp.ClientSession, model : str, url : str, models_results : dict, payload : dict):
    async with session.post(url=url, data=payload) as resp:
        logger.info('Asking for: '+ url)       
        results = await (resp.json())
        results = results['timeseries_containers']
        models_results[model]= results


async def schedule_coroutines(models: list, payload : dict) -> dict:
    
    async with aiohttp.ClientSession() as session:
        models_results = dict.fromkeys(models, {})
        async_requests=[call_predictor(session, 
                                        model, 
                                        url = str(predictor_address+model.lower()),
                                        payload = payload,
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

            logger.info('data received')

        except ValueError as e:
            print(e)
            return 'Error in contacting the data ingestion module'

            
        try:
            payload = {
                "param_config" : json.dumps(param_config),
                "dataset" : ingestion_resp['dataset']
                }

            results = asyncio.run(schedule_coroutines(models, payload))

        except ValueError as e:
            print(e)
            return 'Error in contacting the predictor'


        return {"models_results" : results}, 200


api.add_resource(Manager, '/predict')

timex_manager_address = '127.0.0.1'
timex_manager_port = 6000

if __name__ == '__main__':
    app.run(host=timex_manager_address,
            port=timex_manager_port, debug=True)
