#!/bin/bash

files=("data_ingestion_deployment.yaml" "manager_validator_deployment.yaml" "predictor_deployment.yaml" "timex_app_deployment.yaml")
help_message='Try --help to see the available options.'

if [ ! -z $1 ]; then
    case "$1" in

    --apply-all) 
    for file in ${files[@]}; do 
        kubectl apply -f $file 
    done    
    ;;

    --delete-all) 
    for file in ${files[@]}; do 
        kubectl delete -f $file 
    done    
    ;;

    --help)
    printf ' --apply-all \t\t Apply all the .yaml to build timex \n --delete-all \t\t Delete all the .yaml file used to build timex'
    ;;

    *) 
    echo 'Option not recognized.' $help_message
    ;;
    esac
else
    echo $help_message
fi