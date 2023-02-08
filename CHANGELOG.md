# Change Log

## [1.2.2] - 2022-02-09
- Bump dependencies
- Managed different import mechanisms for Dash

## [1.2.1] - 2021-06-23
- Removed automatic Seasonal part of ARIMA; it could take a long time. Further investigation on `pmdarima` is requested;
- Fixed a bug in which CSV with dates in the format `1959-01` were not correctly recognized;
- Fixed a bug in Yeo-Johnson transformation which would "lose" the datetime index of transformed data;
- Bump dependencies.

## [1.2.0] - 2021-04-08

### Added
- Exponential Smoothing model, with automatic selection of seasonality and inner model
- Options to set max and min values for predictions (if upper/lowerbounds are known beforehand)
- Option to round the predictions to the nearest integer
- Added `_all` key to set additional regressors for all the time-series in a dataset

### Changed
- Multithreading now uses Joblib: it should work on all platforms, and also for NeuralProphet/LSTM models
- Fix bug: cross-correlation were wrong if max_lags was higher than the length of the dataset
- Models can be specified in the `param_config` dictionary using lowercase names
- Bump dependencies

## [1.1.0] - 2021-03-08

### Added

- Resizable histograms in data visualization
- Historical error plot and histogram with percentual 
- Buttons to quickly change box and aggregate box plots period
- Added standard deviation of error in error metrics
- New options for `model_parameters`: `min_values`/`max_values` which can be used to specify minimum and maximum
values to use in prediction
- New option for `model_parameters`: `round_to_integer` useful to point columns'names which should be predicted
as integer numbers

### Changed

- When using `max_threads: 1` multiprocessing is completely ignored, this temporarily fix some issues on MacOs. 
Multiprocessing should be rewritten to work on all platforms.

## [1.0.0] - 2021-02-17

Initial release.