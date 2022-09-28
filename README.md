# TimexDocker

## **Components**:

1. **The [Redpanda](https://redpanda.com/) streaming data platform**
2. **data_ingestion_server**
3. **prediction_server**
4. **validation_server**

## **Project general dependencies**

1. **The [TIMEX](https://github.com/AlexMV12/TIMEX)** library on which the various modules are based on
2. **The [confluent kafka](https://docs.confluent.io/platform/current/clients/confluent-kafka-python/html/index.html#)** library to communicate with the redpanda APIs

The general dependencies poetry configuration files have been created. In order to install it, it is recommend to limit the RAM usage. The command is:
`systemd-run --scope -p MemoryLimit=<desired_ram_usage_limit> ./dependencies-update_command.sh`

## **Redpanda Container Run**

- Install docker and run the command:
  ```docker run -d --pull=always --name=redpanda-1\ -p 9092:9092 \ -p 9644:9644 \ docker.redpanda.com/vectorized/redpanda:latest \ redpanda start \ --overprovisioned \ --smp 1  \ --memory 1G \ --reserve-memory 0M \ --node-id 0 \ --check=false```
  in order to run your red-panda broker instance.
- Open a shell inside the red-panda container and run the command `rpk topic create control_topic` in order to create the first needed channel to start the workflow

## **Docker configuration for NVIDIA GPU Accelerated Containers**

Starting from Docker version 19.03, NVIDIA GPUs are natively supported as Docker devices. 
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is the recommended way of running containers that leverage NVIDIA GPUs. 
Once installed Nvidia-container-toolkit, let's run the container adding the argument `--gpus <how_many_gpus>`. 
For more information [visit here](https://wiki.archlinux.org/titleDocker#Run_GPU_accelerated_Docker_containers_with_NVIDIA_GPUs).

## **Minikube**

NOTE: it's needed to configure minikube to use the kvm2 driver for virtualization, otherwise hardware acceleration would not work. For more information [look here](https://minikube.sigs.k8s.io/docs/drivers/kvm2/). \

### **Containers Usage**

We have basically two main methods:

1. After having built the images, we upload them to minikube 
   `minikube image load <image_name>`
   Set the imagePullPolicy to Never, otherwise Kubernetes will try to download the image.
2. we can directly build the image inside minikube: 
   As the [README](https://github.com/kubernetes/minikube/blob/0c616a6b42b28a1aab8397f5a9061f8ebbd9f3d9/README.md#reusing-the-docker-daemon) describes, you can reuse the Docker daemon from Minikube with eval $(minikube docker-env). So to use an image without uploading it, you can follow these steps:

- Set the environment variables with `eval $(minikube docker-env)`
- Build the image with the Docker daemon of Minikube (e.g. `docker build -t my-image .`)
- Set the image in the pod spec like the build tag (e.g. my-image)
  *Important notes*: You have to run eval $(minikube docker-env) on each terminal you want to use, since it only sets the environment variables for the current shell session. Also in this case, set the imagePullPolicy to Never, otherwise Kubernetes will try to download the image.

3. we can deploy a docker registry as explained [here](https://docs.docker.com/registry/deploying/).
   In the deployment configuration file we need now to specify `<registry_ip_address>/<image_name>`

**NOTE**: the second method is preferred for developing purposes because one can leverage the docker cache when building the images
