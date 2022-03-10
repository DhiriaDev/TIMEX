When running the docker created docker container, please run it using the host network. The command is:

docker run -p 6000:6000 -d --network="host"  timex_manager