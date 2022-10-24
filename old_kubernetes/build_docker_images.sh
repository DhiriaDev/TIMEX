#!/bin/bash

# checking if the docker daemon is active
ps auxw | grep -P '\bdocker(?!-)\b' > /dev/null
if [ $? != 0 ]; then
    echo "The docker daemon is not active. Please, insert your password to activate it."
    read -p "Password:" -s pswd
    echo "$pswd" | sudo -S systemctl start docker
    ps auxw | grep -P '\bdocker(?!-)\b' > /dev/null
    if [ $? == 0 ]; then
        echo "Docker successfully activated"
    else
        echo "There was an error activating docker."
        exit 1
    fi
fi


echo "--------- BUILDING DOCKER IMAGES ---------\n"

echo "Building timex_utils docker image.."
cd utils
DOCKER_BUILDKIT=1 docker build -t timex_utils .

echo "\n\nBuilding timex_data_ingestion docker image.."
cd ../data_ingestion_server
DOCKER_BUILDKIT=1 docker build -t timex_data_ingestion .

echo "\n\nBuilding timex_data_ingestion docker image.."
cd ../timex_app
DOCKER_BUILDKIT=1 docker build -t timex_app .

echo "\n\nBuilding timex_manager docker image.."
cd ../timex_manager
DOCKER_BUILDKIT=1 docker build -t timex_manager .

echo "\n\nBuilding timex_prediction_server docker image.."
cd ../prediction_server
read -p "Please insert password: " -s pswd
echo "$pswd" | sudo -S systemd-run --scope -p MemoryLimit=7000M ./building_command.sh

echo "\n\nBuilding timex_validation_server docker image.."
cd ../validation_server
DOCKER_BUILDKIT=1 docker build -t timex_validation_server .

echo "\n\nCONTAINERS SUCCESSFULLY BUILT!"


read -p "\nDo you want to push the images on the dhiria_repo [Y / n]?" choice
if [ choice == 'n']; then
    exit 0
else
    printf "\n\n--------- PUSHING IMAGES ---------\n"
    docker tag timex_data_ingestion 485540626268.dkr.ecr.eu-west-1.amazonaws.com/dhiria_repo:timex_data_ingestion

    docker tag timex_prediction_server 485540626268.dkr.ecr.eu-west-1.amazonaws.com/dhiria_repo:timex_prediction_server

    docker tag timex_app 485540626268.dkr.ecr.eu-west-1.amazonaws.com/dhiria_repo:timex_app
    docker tag timex_manager 485540626268.dkr.ecr.eu-west-1.amazonaws.com/dhiria_repo:timex_manager

    docker tag timex_validation_server 485540626268.dkr.ecr.eu-west-1.amazonaws.com/dhiria_repo:timex_validation_server


    aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 485540626268.dkr.ecr.eu-west-1.amazonaws.com    

    docker push 485540626268.dkr.ecr.eu-west-1.amazonaws.com/dhiria_repo:timex_data_ingestion
    docker push 485540626268.dkr.ecr.eu-west-1.amazonaws.com/dhiria_repo:timex_manager
    docker push 485540626268.dkr.ecr.eu-west-1.amazonaws.com/dhiria_repo:timex_app
    docker push 485540626268.dkr.ecr.eu-west-1.amazonaws.com/dhiria_repo:timex_validation_server
    docker push 485540626268.dkr.ecr.eu-west-1.amazonaws.com/dhiria_repo:timex_prediction_server

    echo "\n\nCONTAINERS SUCCESSFULLY PUSHED!"
    exit 0
fi
