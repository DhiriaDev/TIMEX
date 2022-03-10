# TimexDocker
## Dockerization of the [Timex app](https://github.com/AlexMV12/TIMEX), a Time series forecasting a.a.S application.

### Components:
1. **timex_app**:\
    the very webapp which deals with the user (uploading a json config file) and send the users requests to the timex_manager. Implemented using the Dash python library.
2. **timex_manager**:\
   it manages the requests coming from the timex_app:
   - sending the request to the data_ingestion_server to load the dataset
   - sending the dataset to the prediction server
   - sending the model(s) results coming from the prediction server to the validation server
   - sending back the best model and its results to the timex_app for the visualization
3. **data_ingestion_server**:\
    after having received the configuration file of the user, it downloads the dataset from the specified URL, creates a DataFrame, performing also some operation on it (such as interpolation to fill empty data, if any), and it responds to the requests sending in the body the final dataset.
4. **prediction_server**:\
    it receives the dataset and the configuration file, it computes the prediction using the chosen algorithm(s)
5. **validation_server**:\
    it receives a set of {model , predictions} for a given time series and it performs the validation, returning the model reaching the best performance

### Docker Images build
The first image to be built is the "timex_utils".
In fact, that image will be the *base image* for the validator_server, the prediction_server and the timex_app. In fact, these three modules share some of the functions contained in the utils module.

To build the images:\
1. enter the directory of the module to build, where a Dockerfile will be found
2. run the following command: `DOCKER_BUILDKIT=1 docker build -t <image_name> .`

Notice: when building the image for the prediction server, it is better to use a swap area/file or to limit the RAM usage of the building process. Some libraries are RAM-eager at compilation time. One way to do that is to exploit the *building_command.sh* file in the prediction_server folder: \
`systemd-run --scope -p MemoryLimit=<desired_ram_usage_limit> ./building_command.sh`
### Docker Containers Run
Example of running the containers. The ports and the host network are needed.

1. **timex_app** :\
    `docker run -p 5000:5000 -d --network="host" timex_app` \
    it is important to specify the network in order to leverage the localhost for making requests. \
    For more info https://docs.docker.com/network/host/
2. **timex_prediction_server**: \
   `docker run -p 3000:3000 -d timex_prediction_server`
3. **timex_data_ingestion**: \
    `docker run -p 4000:4000 -d timex_data_ingestion`
4. **timex_manager**: \
   `docker run -p 6000:6000 -d --network="host"  timex_manager` 
5. **timex_validation_server**: \
   `docker run -p 7000:7000 -d timex_validation_server`

