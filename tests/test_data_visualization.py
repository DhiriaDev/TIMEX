import pickle
import pytest
from timexseries.data_visualization import create_timeseries_dash_children

from timexseries.data_prediction import create_timeseries_containers

from timexseries.data_ingestion import ingest_timeseries

# Testing data visualization is tricky.
# Pre-computed Dash children are saved in a sample pkl file; these visualizations have been checked by a human.
# Hence, a rudimental way to test data visualization is to check that the TIMEX code actually produces those plots,
# by comparing the fresh created ones with the ones saved before.

# Cant test cross-correlation graphs because they are random initialized.


@pytest.fixture
def dash_children(tmp_path):
    param_config = {
        "activity_title": "Example",
        "verbose": "INFO",
        "input_parameters": {
            "source_data_url": "test_datasets/data_visualization/test1.csv",
            "datetime_column_name": "ind",
            "index_column_name": "ind",
            "frequency": "D",
        },
        "model_parameters": {
            "validation_values": 3,
            "delta_training_percentage": 30,
            "forecast_horizon": 5,
            "possible_transformations": "none,log_modified",
            "models": "mockup",
            "main_accuracy_estimator": "mae"
        },
        "historical_prediction_parameters": {
            "initial_index": "2000-01-15",
            "save_path": f"{tmp_path}/historical_predictions.pkl"
        },
        "visualization_parameters": {
            "xcorr_graph_threshold": 0.8,
            "box_plot_frequency": "1W"
        }
    }

    ingested_dataset = ingest_timeseries(param_config)[2]
    timeseries_containers = create_timeseries_containers(ingested_dataset, param_config)

    # Data visualization
    children_for_each_timeseries = [{
        'name': s.timeseries_data.columns[0],
        'children': create_timeseries_dash_children(s, param_config)
    } for s in timeseries_containers]

    return children_for_each_timeseries


@pytest.fixture
def expected_children():
    with open("test_datasets/data_visualization/expected_children1.pkl", "rb") as file:
        expected_children = pickle.load(file)

    return expected_children


def test_1(dash_children, expected_children):
    for computed, expected in zip(dash_children[0]['children'], expected_children[0]['children']):
        assert computed.__str__() == expected.__str__()










