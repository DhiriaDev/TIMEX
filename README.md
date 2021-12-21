# TimexDocker
## Dockerization of the [Timex app](https://github.com/AlexMV12/TIMEX), a Time series forecasting a.a.S application.

***The whole system has been divided into three independent modules***:
1. **timex_app**:\
    the very webapp which deals with the user (uploading a json config file) and dispatches requests by means of REST Apis to both the data ingestion and the data prediction modules. Implemented using the Dash python library.
2. **data_ingestion_server**:\
    after having received the configuration file of the user from the webapp module, it downloads the dataset from the specified URL, creates a DataFrame, performing also some operation on it (such as interpolation to fill empty data, if any), and it responds to the requests sending in the body the final dataset.
3. **data_prediction_server**:\
    it receives the dataset and the configuration file, it computes the prediction using the chosen algorithm, and it sends the result (in the response) to the timex_app. 

***Some technical notes***:\
The json file is sent as a a normal string in POST requests, whereas datasets and results are first serialized in the pickle format, then encoded using the base64 encoder to preserve the structure when transformed into a string to comply with the standard of the http protocol.\
The application is deployed using docker-compose. In every folder, one for each module, there's a docker file which will be used by the docker daemon to generate the docker image for the related module. The dependencies are updated using the python poetry library.\
Since the data prediction module can store the so-called "historical predictions", it could be useful, for the future, to implement a sort of database, linkable to the app using docker-compose, in which a history for each user/dataset is preserved, in order to speed up the elaboration of future requests of the same time, coming from whatever user.

***Application deploy***:
- Create a new docker network. The default one of docker can also be used.\
 In the docker-compose.yml it is specified a network called timex_net. Creating/using a network with a different name means that the .yml file must be updated.
- The command to run the app is:\
*docker-compose up --build*\
Note that the argument *--build* is necessary just the first time, since the images and containers must be built. Then, it can be discarded, if one doesn't want to rebuild the whole system at every run.