import sys, logging, os
import json, copy

from utils import *
from multiprocessing import Process
from confluent_kafka import *

kafka_address = '0.0.0.0:9092'
control_topic = 'manager_control'
client_id = 'timex_manager'
group_id = 'timex_manager'


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

def parse_source_data_url(data_url : str) -> str :
    if 'drive.google.com' in data_url:
        return 'https://drive.google.com/uc?export=download&confirm=t&id=' + \
                        data_url.split('/')[-2]
    else:
        return data_url

def prepare_params(param_config:dict, models : list) -> dict :

    configurations_for_each_model = dict.fromkeys(models, {})
    hist_paths = []
    

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

        dataset_url = param_config['input_parameters']['source_data_url']
        param_config['input_parameters']['source_data_url'] = parse_source_data_url(dataset_url)

        # First, we check if the requested model(s) is(are) valid.
        # If not, the request is not even forwarded to the data ingestion module
        models = [*param_config["model_parameters"]["models"].split(",")]
        if not all(item in available_predictors for item in models):
            logger.error('One or more of the requested models is not valid')
            return 400


        try:
            logger.info('Contacting the data ingestion module')
            # here data has been sent to the data ingestion module
            ingestion_resp = json.loads(
                requests.post(data_ingestion_address,
                    data={'param_config' : json.dumps(param_config)}).text
                    )

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








def call_data_ingestion(param_config : str) :
    
    data_ingestion_control_topic = 'ingestion_control'
    conf = {
        "bootstrap.servers" : kafka_address, 
        "client.id": client_id, 
        "acks":1,
        "retries":3,
        "batch.size":1,
        "max.in.flight.requests.per.connection":1
    }

    producer = Producer(conf)

    headers = {'prod_id' : str(client_id),
               'type' : str(MessageType.control_message.value)}

    data = param_config

    producer.produce(topic = ingestion_topic,
                        value = json.dumps(data).encode('utf-8'),
                        headers = headers)
    producer.flush()


if __name__ == "__main__" :
    timex_watcher_config = {"bootstrap.servers" : kafka_address,
                            "client.id" : client_id,
                            "group.id" : group_id,
                            "max.in.flight.requests.per.connection" : 1,
                            "auto.offset.reset":'earliest'
                            }
    works_to_do = [call_data_ingestion]

    watcher = Watcher(config_dict= timex_watcher_config, works_to_do= works_to_do)
    watcher.listen_on_control(control_topic = control_topic)
