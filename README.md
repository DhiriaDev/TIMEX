# TimexDocker
## Dockerization of the [Timex app](https://github.com/AlexMV12/TIMEX), a Time series forecasting a.a.S application.

***The whole system has been divided into five independent modules***:
1. **timex_app**:\
    the very webapp which deals with the user (uploading a json config file) and send the users requests to the timex_manager. Implemented using the Dash python library.
2. **timex_manager**:\
   it manages the requests coming from the timex_app:
   - sending the request to the data_ingestion_server to load the dataset
   - sending the dataset to the prediction server
   - sending the model(s) results coming from the prediction server to the validation server
   - sending back the best model and its results to the timex_app for the visualization
3. **data_ingestion_server**:\
    after having received the configuration file of the user, it downloads the dataset from the specified URL, creates a DataFrame, performing also some operation on it (such as interpolation to fill empty data, if any), and it responds to the requests sending in the body the final dataset.
4. **prediction_server**:\
    it receives the dataset and the configuration file, it computes the prediction using the chosen algorithm(s)
5. **validation_server**:\
    it receives a set of {model , predictions} for a given time series and it performs the validation, returning the model reaching the best performance

