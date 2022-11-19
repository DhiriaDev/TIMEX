#!/bin/bash

for i in {1..3}; do
    ( python3 -m locust -f tests/locust_files/locustfile_fra.py --worker "zk1.dhiria.com:9092, zk2.dhiria.com:9092, zk4.dhiria.com:9092" "./dataset_examples/ElectricityLoad/configElectricityLoad.json" "/home/eks-timex/shared/dhiria-shared/redpanda-tests/data_to_send/ElectricityLoad.csv" & )
done

python3 -m locust -f tests/locust_files/locustfile_fra.py --master --expect-workers=3 --web-host=127.0.0.1 "zk1.dhiria.com:9092, zk2.dhiria.com:9092, zk4.dhiria.com:9092" "./dataset_examples/ElectricityLoad/configElectricityLoad.json" "/home/eks-timex/shared/dhiria-shared/redpanda-tests/data_to_send/ElectricityLoad.csv"
