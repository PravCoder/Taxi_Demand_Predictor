from datetime import datetime, timedelta

import hopsworks
from hsfs.feature_store import FeatureStore
import pandas as pd
import numpy as np

import src.config as config

def get_hopsworks_project():
    # login to hopsworks using project name/api key
    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )

def get_feature_store():
    project = get_hopsworks_project()  # return pointer to feature store
    return project.get_feature_store()

def get_model_predictions(model, features):
    # given model-obj, features, pass features into model generate predictions
    predictions = model.predict(features)
    # create empty df
    results = pd.DataFrame()
    # set results-df-col-location-id equal to the features-df-col-location-id
    results["pickup_location_id"] = features["pickup_location_id"].values
    # set results-df-col-predicted-demand is equal to the predictions-df
    results["predicted_demand"] = predictions.round(0)
    # how do we get these features? look below
    return results

# GIVEN CURRENT-DATE
def load_batch_of_features_from_store(current_date):
    feature_store = get_feature_store()  # connects to feature-store

    n_features = config.N_FEATURES

    # read time-series data from the feature store in hopsworks
    fetch_data_to = current_date - timedelta(hours=1) # calculate dates to fetch data from and to
    fetch_data_from = current_date - timedelta(days=28)
    print(f"Fetching data from {fetch_data_from} to {fetch_data_to}")
    # get the feature-store's feature view using view-name and veresion
    feature_view = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_NAME,
        version=config.FEATURE_VIEW_VERSION
    )
    # get time-series data by calling get-batch-data passing in start-time and end-time
    ts_data = feature_view.get_batch_data(
        start_time=(fetch_data_from - timedelta(days=1)),
        end_time=(fetch_data_to - timedelta(days=1))
    )
    # from ts-data-df select only rows whose pickup-hour is between the from-to dates
    ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, fetch_data_to)]
    print(f"{ts_data=}")
    # from ts-data-df from the pikcup-locaiton-id-col get the unqiue values
    location_ids = ts_data["pickup_locaation_id"].unique()

    # create numpy-arr whose shape is (num of locations, num of features)
    x = np.ndarray(shape=(len(location_ids), n_features), dtype=np.float32)
    # iterate through every unique location-id
    for i, location_id in enumerate(location_ids):
        # ts-df get the rows whose col-pickup_location_id is equal to cur-location-id
        ts_data_i = ts_data.loc[ts_data.pickup_location_id == location_id, :]
        # sort the rows by pickup hour-col
        ts_data_i = ts_data_i.sort_values(by=["pickup_hour"])
        # set numpy-arr ith element equal to ts-df rides-col
        x[i, :] = ts_data_i["rides"].values

    # create features-df passing in x-np-arr, columns are iterate through the number of features
    features = pd.DataFrame(
        x, 
        columns=[f"rides_previous_{i+1}_hour" for i in reversed(range(n_features))]
    )

    features["pickup_hour"] = current_date
    features["pickup_location_id"] = location_ids
    return features

def load_model_from_registry():
    import joblib
    from pathlib import Path

    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    model = model_registry.get_model(
        name=config.MODEL_NAME,
        version=config.MODEL_VERSION,
    )

    model_dir = model.download()
    
    # get model from hopsworks model registry
    model = joblib.load(Path(model_dir) / "model.pkl")

    return model
