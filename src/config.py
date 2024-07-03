import os
from dotenv import load_dotenv
from src.paths import PARENT_DIR


load_dotenv(PARENT_DIR / ".env")

HOPSWORKS_PROJECT_NAME = "taxi_demand_pravachan"
try:
    HOPSWORKS_API_KEY = os.environ["HOPSWORKS_API_KEY"]  # extract the api-key-var from the file
except:
    raise Exception("Create an .env file on projedt root with the hopsworks-api-key")

FEATURE_GROUP_NAME = "time_series_hourly_feature_group"
FEATURE_GROUP_VERSION =  3 

FEATURE_VIEW_NAME = "time_series_hourly_feature_view"
FEATURE_VIEW_VERSION = 1