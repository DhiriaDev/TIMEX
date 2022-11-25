echo "Test Begins"

while true;
do
  curl -s "http://localhost:8089/stats/requests" >> locust-stats.json;
  echo "" >> locust-stats.json;
  sleep 2;
done
