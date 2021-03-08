# Change Log

## [1.1.0] - 2020-03-08

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

## [1.0.0] - 2020-02-17

Initial release.