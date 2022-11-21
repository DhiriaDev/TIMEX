#!/bin/bash

# checking if the docker daemon is active
#ps auxw | grep -P '\bdocker(?!-)\b' > /dev/null
#if [ $? != 0 ]; then
#    echo "The docker daemon is not active. Please, insert your password to activate it."
#    read -p "Password:" -s pswd
#    echo "$pswd" | sudo -S systemctl start docker
#    ps auxw | grep -P '\bdocker(?!-)\b' > /dev/null
#    if [ $? == 0 ]; then
#        echo "Docker successfully activated"
#    else
#        echo "There was an error activating docker."
#        exit 1
#    fi
#fi


echo "--------- BUILDING DOCKER IMAGES ---------\n"

echo "Building baseImage.."
DOCKER_BUILDKIT=1 docker build -t baseimage --file ./docker/baseImage_dockerfile ./
if [ $? != 0 ]; then
    echo "baseImage failed"
    exit 1
fi

echo "Building validation docker image.."
DOCKER_BUILDKIT=1 docker build -t validation_container --file ./docker/validation_dockerfile ./
if [ $? != 0 ]; then
    echo "validation failed"
    exit 1
fi

echo "\n\nBuilding ingestion docker image.."
DOCKER_BUILDKIT=1 docker build -t ingestion_container --file ./docker/ingestion_dockerfile ./
if [ $? != 0 ]; then
    echo "ingestion failed"
    exit 1
fi

echo "\n\nBuilding prediction docker image.."
DOCKER_BUILDKIT=1 docker build -t prediction_container --file ./docker/prediction_dockerfile ./
if [ $? != 0 ]; then
    echo "prediction failed"
    exit 1
fi

echo "\n\nCONTAINERS SUCCESSFULLY BUILT!"


read -p "\nDo you want to push the images on the dhiria_repo [Y / n]?" choice
if [$choice = "n"]; then
    exit 0
else
    printf "\n\n--------- PUSHING IMAGES ---------\n"
    docker tag validation_container 485540626268.dkr.ecr.eu-west-1.amazonaws.com/dhiria_repo:validation_container

    docker tag ingestion_container 485540626268.dkr.ecr.eu-west-1.amazonaws.com/dhiria_repo:ingestion_container

    docker tag prediction_container 485540626268.dkr.ecr.eu-west-1.amazonaws.com/dhiria_repo:prediction_container
    

    aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 485540626268.dkr.ecr.eu-west-1.amazonaws.com    

    docker push 485540626268.dkr.ecr.eu-west-1.amazonaws.com/dhiria_repo:validation_container
    docker push 485540626268.dkr.ecr.eu-west-1.amazonaws.com/dhiria_repo:ingestion_container
    docker push 485540626268.dkr.ecr.eu-west-1.amazonaws.com/dhiria_repo:prediction_container

    echo "\n\nCONTAINERS SUCCESSFULLY PUSHED!"
    exit 0
fi
