import sys
import logging

from flask import Flask, request
from flask_restful import Api, Resource
import json
import asyncio, aiohttp

app = Flask(__name__)
api = Api(app)

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


predictor_address = 'http://127.0.0.1:3000/'
available_predictors = ['arima', 'fbprophet', 'lstm', 'mockup', 'exponentialsmoothing']

async def call_predictor(session : aiohttp.ClientSession, model : str, url : str, models_results : dict):
    async with session.post(url=url, data=request.form) as resp:
        logger.info('asking for: '+url)       
        results = await (resp.json())
        results = results['timeseries_containers']
        models_results[model]= results


async def schedule_coroutines(models) -> dict:
    async with aiohttp.ClientSession() as session:
        models_results = dict.fromkeys(models, {})
        async_requests=[call_predictor(session, 
                                        model, 
                                        url = str(predictor_address+model.lower()),
                                        models_results=models_results) for model in models]
        await asyncio.gather(*async_requests)
        return models_results


class Orchestrator(Resource):

    def post(self):
        
        logger.info('Request received')

        param_config = json.loads(request.form['param_config'])

        models = [*param_config["model_parameters"]["models"].split(",")]

        # Let's check if the requested model(s) is(are) valid
        if not all(item in available_predictors for item in models):
            logger.error('One or more of the requested models is not valid')
            return 400

        results = asyncio.run(schedule_coroutines(models))
        payload = {"models_results" : results}

        return payload, 200


api.add_resource(Orchestrator, '/predict')

orchestrator_address = '127.0.0.1'
orchestrator_port = 6000

if __name__ == '__main__':
    app.run(host=orchestrator_address,
            port=orchestrator_port, debug=True)
