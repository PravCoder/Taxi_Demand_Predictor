import sys
import os
import importlib
script_dir = os.getcwd()  
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

# BREAK
import zipfile 
from datetime import datetime, timedelta

import requests
import numpy as np
import pandas as pd
# plotting libraries
import streamlit as st
import geopandas as gpd
import pydeck as pdk
# from inference.py import 2 funcs
from inference import (
    # load_predictions_from_store,
    load_batch_of_features_from_store,
    load_model_from_registry,
    get_model_predictions
)
from paths import DATA_DIR
from plot import plot_one_sample
import pytz


st.set_page_config(layout="wide")  # set page layout to wide
# get current date and set title and header as cur-date
current_date = pd.to_datetime(datetime.now(pytz.utc)).floor("H").replace(tzinfo=None)
st.title(f"Taxi Demand Predictor")
st.header(f"{current_date}")

# add header to sidebar
progress_bar = st.sidebar.header("o Working Progress")
progress_bar = st.sidebar.progress(0)
N_STEPS = 7


def load_shape_data_file():
    # url from taxi website
    URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
    response = requests.get(URL)  # send get-request to url
    path = DATA_DIR / f"taxi_zones.zip"  # path is the data directory adn zip-file
    # if response is successful, open the path and write the content of the response which is the data into the file specififed in the path
    if response.status_code == 200:
        open(path, "wb").write(response.content)
    else:
        raise Exception(f"{URL} is not avaible")
    # unzip the file, passing in path
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR  / "taxi_zones")
    
    # load and return as geopandas data-frame
    return gpd.read_file(DATA_DIR / 'taxi_zones/taxi_zones.shp').to_crs('epsg:4326')

with st.spinner(text="Downloading shape file to plot taxi zones"):
    geo_df = load_shape_data_file() # doanlod the shape file returns geo-df
    st.sidebar.write("Shape file was downlaoded")
    progress_bar.progress(1/N_STEPS)

with st.spinner(text="Fetching model predictions from the store"):
    # features-df = get most recent features stored by feature-pipeline from feature-store
    features = load_batch_of_features_from_store(current_date)
    st.sidebar.write('✅ Model predictions arrived') # update UI
    progress_bar.progress(2/N_STEPS)

with st.spinner(text="Loading ML model from model registry"):
    # model-obj = from the model registyr in hopsworks get teh saved model that was saved by model-training-pipeline
    model = load_model_from_registry()  
    st.sidebar.write("ML model was loaded from the registry") # update UI
    progress_bar.progress(3/N_STEPS)

with st.spinner(text="Computing Model predictions"):
    # given model-obj and features-df calls predict on model passing features
    results = get_model_predictions(model, features)
    st.sidebar.write("Model predictions arrived") # update UI
    progress_bar.progress(4/N_STEPS)