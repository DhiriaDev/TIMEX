import subprocess


if __name__ == '__main__':
    kafka_address = '0.0.0.0:9092'

    subprocess.Popen(["python tests/ingestion_instance.py --kafka_address " + kafka_address], shell=True)
    subprocess.Popen(["python tests/prediction_instance.py --kafka_address " + kafka_address], shell = True)
    subprocess.Popen(["python tests/validation_instance.py --kafka_address " + kafka_address], shell = True)