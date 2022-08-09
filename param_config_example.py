param_config = {
    "activity_title": "job_name", # hashlib.md5(string= (path+str(prod_id)).encode('utf-8')).hexdigest()
    "verbose": "INFO",
    "input_parameters": {
        "source_data_url": "https://drive.google.com/file/d/1buo6gncSsEcK9nyi7iP--hKT9W44sWph/view?usp=sharing",
        "columns_to_load_from_url": "Date,Close",
        "datetime_column_name": "Date",
        "index_column_name": "Date",
        "frequency": "D"
    },
    "model_parameters": {
        "test_values": 15,
        "delta_training_percentage": 15,
        "prediction_lags": 10,
        "possible_transformations": "none,log_modified",
        "models": "fbprophet",
        "main_accuracy_estimator": "rmse"
    },
    "visualization_parameters": {
        "xcorr_graph_threshold": 0.8,
        "box_plot_frequency": "1W"
    }
}