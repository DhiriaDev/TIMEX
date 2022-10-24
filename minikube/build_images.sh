#!/usr/bin/env bash

cd ../
docker build -t prediction:latest -f docker/prediction_dockerfile .
