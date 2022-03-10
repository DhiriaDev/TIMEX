#!/bin/bash

<<COMMENT
    It is better to run the command below using first:
    systemd-run --scope -p MemoryLimit=<desired_ram_usage_limit> ./building_command.sh
COMMENT

DOCKER_BUILDKIT=1 docker build -t timex_prediction_server .