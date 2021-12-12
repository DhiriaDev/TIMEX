# TimexDocker
dockerization of the Timex app

building command: 
    " sudo DOCKER_BUILDKIT=1 docker build --tag timex-docker .  "

running command:
    " sudo docker run -p 5000:80 timex-docker  "