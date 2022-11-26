rm ./stats-50usrs_slow.json

curl -X POST -H "application/x-www-form-urlencoded; charset=UTF-8" -d "user_count=50&spawn_rate=0.033&host=&run_time=&kafka_address=zk1.dhiria.com%3A9092%2C+zk2.dhiria.com%3A9092%2C+zk4.dhiria.com%3A9092&param_config=.%2Fdataset_examples%2FElectricityLoad%2FconfigElectricityLoad.json&file_path=%2Fhome%2Feks-timex%2Fshared%2Fdhiria-shared%2Fredpanda-tests%2Fdata_to_send%2FElectricityLoad.csv" http://localhost:8089/swarm

runtime="35 minute"
endtime=$(date -ud "$runtime" +%s)

echo "Test Begins"

while [[ $(date -u +%s) -le $endtime ]];
do
  curl -s "http://localhost:8089/stats/requests" >> stats-50usrs_slow.json;
  echo "" >> stats-50usrs_slow.json;
  sleep 2;
done

curl "http://localhost:8089/stop"