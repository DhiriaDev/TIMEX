import subprocess


if __name__ == '__main__':
    subprocess.Popen(["python tests/ingestion_instance.py"], shell=True)
    subprocess.Popen(["python tests/prediction_instance.py"], shell = True)
    subprocess.Popen(["python tests/validation_instance.py"], shell = True)