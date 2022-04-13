# TimexDocker
## Dockerization of the [Timex app](https://github.com/AlexMV12/TIMEX), a Time series forecasting a.a.S application.

## **Components**:
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
For further information, open the [Timex power point](Documents/Timex.pptx)

## **Project general dependencies**
In addition to the dependencies of each submodule, also the general dependencies poetry configuration files have been created. In order to install it, it is recommend to limit the RAM usage. The command is:
`systemd-run --scope -p MemoryLimit=<desired_ram_usage_limit> ./dependencies-update_command.sh`
When running the servers outside a docker container, you need to add to the shell: `$PYTHONPATH:$(pwd)/utils:$(pwd)/prediction_server/models` before of starting the server.

## **Docker Images build**
### **Note: you can use the create_containers.sh to build and push all the images**

The first image to be built is the "timex_utils".
In fact, that image will be the *base image* for the validator_server, the prediction_server and the timex_app. In fact, these three modules share some of the functions contained in the utils module.

To build the images:\
1. enter the directory of the module to build, where a Dockerfile will be found
2. run the following command: `DOCKER_BUILDKIT=1 docker build -t <image_name> .`

Notice: when building the image for the prediction server, it is better to use a swap area/file or to limit the RAM usage of the building process. Some libraries are RAM-eager at compilation time. One way to do that is to exploit the *building_command.sh* file in the prediction_server folder: \
`systemd-run --scope -p MemoryLimit=<desired_ram_usage_limit> ./building_command.sh`
## **Docker Containers Run**
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

## **Docker configuration for NVIDIA GPU Accelerated Containers**
Starting from Docker version 19.03, NVIDIA GPUs are natively supported as Docker devices. \
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is the recommended way of running containers that leverage NVIDIA GPUs. \
Once installed Nvidia-container-toolkit, let's run the container adding the argument `--gpus <how_many_gpus>`. \
For more information [visit here](https://wiki.archlinux.org/titleDocker#Run_GPU_accelerated_Docker_containers_with_NVIDIA_GPUs).

## **Minikube**
NOTE: it's needed to configure minikube to use the kvm2 driver for virtualization, otherwise hardware acceleration would not work. For more information [look here](https://minikube.sigs.k8s.io/docs/drivers/kvm2/). \

### **Containers Usage**
We have basically two main methods:
1. After having built the images, we upload them to minikube \
    `minikube image load <image_name>`
  Set the imagePullPolicy to Never, otherwise Kubernetes will try to download the image.

2. we can directly build the image inside minikube: \
  As the [README](https://github.com/kubernetes/minikube/blob/0c616a6b42b28a1aab8397f5a9061f8ebbd9f3d9/README.md#reusing-the-docker-daemon) describes, you can reuse the Docker daemon from Minikube with eval $(minikube docker-env). So to use an image without uploading it, you can follow these steps:
  
  - Set the environment variables with `eval $(minikube docker-env)`
  - Build the image with the Docker daemon of Minikube (e.g. `docker build -t my-image .`)
  - Set the image in the pod spec like the build tag (e.g. my-image)
    *Important notes*: You have to run eval $(minikube docker-env) on each terminal you want to use, since it only sets the environment variables for the current shell session. Also in this case, set the imagePullPolicy to Never, otherwise Kubernetes will try to download the image.


3. we can deploy a docker registry as explained [here](https://docs.docker.com/registry/deploying/). 
  In the deployment configuration file we need now to specify `<registry_ip_address>/<image_name>`

**NOTE**: the second method is preferred for developing purposes because one can leverage the docker cache when building the images

## **MODELS VALIDATION**
To perform validation, and thus to choose the best model among those requested, some metrics are taken into account:
- R_squared:\
  It denotes the proportion of the dependent variable’s variance that may be explained by the independent variable’s variance. A high R2 value shows that the model’s variance is similar to that of the true values, whereas a low R2 value suggests that the two values are not strongly related. The most important thing to remember about R-squared is that it does not indicate whether or not the model is capable of making accurate future predictions. It shows whether or not the model is a good fit for the observed values, as well as how good of a fit it is. A high R2 indicates that the observed and anticipated values have a strong association. Moreover, the R2 compares the given model with the simple "average predictor". Thus, having a negative R2 means that the model is worse than a simple average predictor. Ideally, we would like to have R2=1.
- Root Mean Squared Error (RMSE) & Mean Absolute Error (MAE):\
  When comparing forecast methods applied to a single time series the the RMSE and the MAE are very popular since they are straightforward to both compute and understand (the MAE little bit more than the other one).
    - Minimizing the MAE will lead to forecasts of the median
    - Minimizing the RMSE will lead to forecasts of the mean.
    - The greater difference between them, the greater the variance in the individual errors in the sample.
    - If the RMSE=MAE, then all the errors are of the same magnitude
  Notice: RMSE >= MAE.
  In case of multiple time series, the RMSE and the MAE cannot be used as-is anymore, since they are scale-dependent. Other metrics can be used, such as the NormalizedRMSE
- Arithmetic Mean of error (AM)
- Standard deviation of error (SD)