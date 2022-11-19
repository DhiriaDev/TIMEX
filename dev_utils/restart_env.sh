#!/bin/bash

kubectl delete -f docker/setup_cluster.yaml

rpk --brokers='zk1.dhiria.com:9092,zk2.dhiria.com:9092,zk4.dhiria.com:9092' topic delete -r '.*'
rpk --brokers='zk1.dhiria.com:9092,zk2.dhiria.com:9092,zk4.dhiria.com:9092' topic create control_topic -r 3 -p 1

kubectl apply -f docker/setup_cluster.yaml
