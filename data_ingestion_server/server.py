import logging
import sys

from flask import Flask, request
from flask_restful import Api, Resource
import json, pickle, base64

from data_ingestion.data_ingestion import ingest_timeseries




app = Flask(__name__)
api = Api(app)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


class Ingest(Resource):
    def post(self):
        '''
        The current request method is available by using the method attribute.
        To access form data (data transmitted in a POST or PUT request) you can use the form attribute
        '''
        try:
            param_config = json.loads(request.form['param_config'])

            dataset = ingest_timeseries(param_config)

            '''
                Now we need to convert the dataset to the pickle format in order to make it serializable.
                But in order to send it by means of a POST request we need to encode in the base64 format.
                In fact, the 64encoded string is also a bytes-like string,but it doesn't contain the hex escape 
                sequences so when it gets converted to a string, the string is still preserved when encoding it to bytes.
            '''
            dataset = pickle.dumps(dataset)
            dataset = (base64.b64encode(dataset)).decode('utf-8')

            payload = {}
            payload['param_config'] = json.dumps(param_config)
            payload['dataset'] = dataset

            return payload, 200

        except ValueError as err:
            logger.error(err.with_traceback)
            return 400


api.add_resource(Ingest, '/ingest')
    
data_ingestion_address = '127.0.0.1'
data_ingestion_docker_address='0.0.0.0'
data_ingestion_port = 4000

if __name__ == '__main__':
    app.run(host=data_ingestion_docker_address, port=data_ingestion_port ,debug=True)

