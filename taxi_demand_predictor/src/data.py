from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from pdb import set_trace as stop
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.paths import RAW_DATA_DIR, TRANSFORMED_DATA_DIR  # import commonly used paths

# PULLS RAW DATA
def download_one_file_of_raw_data(year, month):  # given a year and month downloads that data
    # this is the url that immediately downloads the data on the nyc-tax-site, pasing placeholders year and month
    URL = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    response = requests.get(URL) # sends http-get-request to that url, returns a response object which contains the servers response to request which is the taxi-data

    if response.status_code == 200: # if response was successful
        path = RAW_DATA_DIR / f'rides_{year}-{month:02d}.parquet'  # contructs a file path inside data folder, raw-folder, specifying file name yaer-month.extension
        open(path, "wb").write(response.content) # open the file specified by the path and write the contents of the response object which is data into it
        return path
    else:
        raise Exception(f"{URL} is not available")
    
# VALIDATE RAW DATA
def validate_raw_data(rides, year, month):  # rides-dataframe, year/month integer of data. given rows with pickup-datetime-column removes those that are outside valid range
    this_month_start = f"{year}-{month:02d}-01" # construct string of date of first day of given month, year-month-day, day is always 1st
    next_month_start = f"{year}-{month+1:02d}-01" if month < 12 else f'{year+1}-01-01'  # construct string of date of first day of next month, 

    rides = rides[rides.pickup_datetime >= this_month_start] # for all rows in rides-df only select the rows whose pickup-datetime-column is after the starting date of given month
    rides = rides[rides.pickup_datetime < next_month_start] # for all rows in rides-df only select the the rwos whose pikcup-datetime-column is before teh starting date of next month

    return rides

# GIVEN A YEAR AND LIST OF MONTHS DOWNLOADS DATA FOR THOSE MONTHS AND VALIDATES IT
def load_raw_data(year, months=None):
    rides = pd.DataFrame()   # initlize empty df
    # semantic checks
    if months is None:
        months = list(range(1, 13))
    elif isinstance(months, int):
        months = [months]
    
    print(months)
    # iterate each month-integer in month-list
    for month in months:
        # construct string of path for cur-month-year raw-data-file, to see if a file of raw data for this specific year and month exists in raw-data-dir
        local_file = RAW_DATA_DIR / f'rides_{year}-{month:02d}.parquet'
        if not local_file.exists():  # if the file doesn't exist
            try:        # call function passing year/month which pulls the data for year/month from website and stores it in raw-directory
                print(f'Downloading file {year}-{month:02d}')
                download_one_file_of_raw_data(year, month)  # if that file doesnt exist for that month/year in raw-data-dir, download raw-data from website and store it in raw-data-dir
            except:    # else file is not there
                print(f"{year}-{month:02d} file is not available")
                continue
        else:  # file for this year-month already exists
            print(f"File {year}-{month:02d} is already in local storage")

        # read in created-raw-data-file for cur-month-year, as a dataframe
        rides_one_month = pd.read_parquet(local_file)

        # for every row only select those two columns
        rides_one_month = rides_one_month[["tpep_pickup_datetime", "PULocationID"]]

        # rename the 2 columns old:new, for cur-month-year-dataframe
        rides_one_month.rename(columns={"tpep_pickup_datetime":"pickup_datetime", "PULocationID":"pickup_location_id"}, inplace=True)

        # call func which removes rows-dates that are not in range, passing dataframe cur-year-month
        rides_one_month = validate_raw_data(rides_one_month, year, month)

        # append to existing data, ther rides-total-dataframe, rides for cur-month
        rides = pd.concat([rides, rides_one_month])

    # if there is no data return empty-df
    if rides.empty:
        return pd.DataFrame()
    # keep only time and origin of ride
    else:
        rides = rides[["pickup_datetime", "pickup_location_id"]]
        return rides



# FOR INSTANCES WHERE THE LOCATION AND HOUR WHERE THERE WERE NO RIDES, THEY DO NOT APPEAR, SE WE WANT TO PLACE THEM WITH ZEROS
def add_missing_slots(agg_rides):
    location_ids = agg_rides['pickup_location_id'].unique() # returns array of unqiue pickup-location-ids
    # date-time-index object representing complete range of hourly time slots from min to max pickup-hour in agg-rides
    full_range = pd.date_range(agg_rides["pickup_hour"].min(), agg_rides["pickup_hour"].max(), freq="H")
    # an empty df to store the final result. 
    output = pd.DataFrame()
    
    # iterate throughn every unqiue location-id
    for location_id in tqdm(location_ids):
        # if df.attribute is equal to cur-location-id, loc selects teh specified columns, aslong as the conditions is met
        # first filters rows by if its equal to cur-location-id, then selects the two columns 
        agg_rides_i = agg_rides.loc[agg_rides.pickup_location_id == location_id, ["pickup_hour","num_of_rides"]]  # return is the filtered data-frame

        agg_rides_i.set_index("pickup_hour", inplace=True) # sets the pickup-hour-col as the index of the df-agg-rides-i
        agg_rides_i.index = pd.DatetimeIndex(agg_rides_i.index)  #setting the index-aattribute equal to DTI-obj
        agg_rides_i = agg_rides_i.reindex(full_range, fill_value=0) # fills any missing values in agg-rides-i with 0
        agg_rides_i["pickup_location_id"] = location_id  # adding new column equal to cur-location-id

        output = pd.concat([output, agg_rides_i]) # mergs cur-agg-rides-i

    output = output.reset_index().rename(columns={"index":"pickup_hour"}) # modifies output-df by restting its index and renaming the newly created column

    return output


def transform_raw_data_into_ts_data(rides):
    # sum rides per location and pickup_hour
    # add column that is the rounded hour, because we want to work with time series data at an hourly frequency
    # rides[new-col] = rides get the datetime columns and round the hour part of it. .dt acess the datetime properties of the pickup_datetime columns
    rides['pickup_hour'] = rides['pickup_datetime'].dt.floor('H')
    # group/count the number of rides per location_ID per pickup_hour
    # groupby(): given the rides-dataframe, groups the 2 columns which means all the rows with the same pickup_hour and pickup_location_id will be grouped together
    # size(): counts the number rows/rides ine ach group, this produces a series wher ehte index is the grouped columns, and the values are the counts of rides in each group, 
    # reset-index(): converts the series back into a dataframe and resets the index
    agg_rides = rides.groupby(['pickup_hour', 'pickup_location_id']).size().reset_index()
    agg_rides.rename(columns={0: 'num_of_rides'}, inplace=True)

    # add rows for (locations, pickup_hours)s with 0 rides
    agg_rides_all_slots = add_missing_slots(agg_rides)

    return agg_rides_all_slots
     


# GIVEN TS-DATA LOOPS THROUGH ALLL LOCATION-IDS AND CREATES EXAMPLES FEAUTRES AND TARGETS, look at video to see how features/targets are sliced
def transform_ts_data_into_features_and_target(ts_data, input_seq_len, step_size):

    location_ids = ts_data["pickup_location_id"].unique()  # get the unqiue locaiton-ids in the columns
    features = pd.DataFrame()  # create empty df
    targets = pd.DataFrame()   # create empty df

    for location_id in tqdm(location_ids):
        # for all the rows in ts-data-df that have pickup-loc-id equal to cur-location-id select its pickup-hour adn num-of-rides columns
        ts_data_one_location = ts_data.loc[ts_data.pickup_location_id == location_id, ["pickup_hour", "num_of_rides"]]

        # get list of indicies-triplet for cur-location passing ts-dataframe for cur-location and input-size 
        indices = get_cutoff_indices(ts_data_one_location, input_seq_len, step_size) 

        n_examples = len(indices)  # get number of examples
        x = np.ndarray(shape=(n_examples, input_seq_len), dtype=np.float32)  # initlize empty array of size (examples, input-nodes) for train-x-data
        y = np.ndarray(shape=(n_examples), dtype=np.float32)                 # initlize empty array 1D of size examples for train-y-data
        pickup_hours = []

        # iterate through all triplet-indices for cur location which we can get all of the examples (features and target) per triplet group
        for i, idx in enumerate(indices):
            # set ith row in x-dataframe equal to the cur-ts-loc-data sliced from start-indx to mid-indx of cur-idx-triplet, get the num-of-rides column, gets features
            x[i, :] = ts_data_one_location.iloc[idx[0]:idx[1]]["num_of_rides"].values
            # set ith row of y-dataframe equal to cur-ts-loc-data sliced from mid-indx to last-indx of cur-idx-triplet, get the num-of-rides column gets targets
            y[i] = ts_data_one_location.iloc[idx[1]:idx[2]]['num_of_rides'].values
            # get the row at position idx[1] of cur-loc-dataframe, select pickup-hour-col and add it to pickup-horus list, this is done for every example-indicies-group for every location 
            pickup_hours.append(ts_data_one_location.iloc[idx[1]]['pickup_hour'])

        # convert x-numpy-arr to data-frame specifying the columns-names of df, iterating through all feature-input-node-indiceis, and creating column in dataframe for each
        features_one_location = pd.DataFrame(x, columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(input_seq_len))])
        # set the pickup-hour-col for cur-location-feature-dataframe equal to the pickup-hours-list
        features_one_location["pickup_hour"] = pickup_hours
        # set the pickup_location_id-col of cur-location-feature-dataframe equal to the cur-location-id
        features_one_location["pickup_location_id"] = location_id

        # convert y-numpy-arr to data-frame specifying the columns-name of df which is just one
        targets_one_location = pd.DataFrame(y, columns=[f"target_rides_next_hour"])

        # concatenate/add the features for cur-location to features-total-df, same for targets
        features = pd.concat([features, features_one_location])
        targets = pd.concat([targets, targets_one_location])

    features.reset_index(inplace=True, drop=True)
    targets.reset_index(inplace=True, drop=True)

    # return features-dataframe and targets-column
    return features, targets["target_rides_next_hour"]

    


# given data-frame, number of features for model, and stpe-size
def get_cutoff_indices(data, n_features, step_size): 
    stop_position = len(data)-1  # stop-pos is last-index-row

    subseq_first_idx = 0            # starting index init at 0th index
    subseq_mid_idx = n_features     # mid-index is the number of features-index-row
    subseq_last_idx = n_features+1  # last-index is one after that which is target
    indicies = []                   # stores triplets of indices for each exmaple (first, mid, last) where first to mid is the features, mid to last is the targets

    # while the last-index has not reached end of df
    while subseq_last_idx <= stop_position:
        # add triplet of indices (a,b,c), a to b is features, b to c is target
        indicies.append((subseq_first_idx, subseq_mid_idx, subseq_last_idx))

        subseq_first_idx += step_size  # update all indices by step-size, to move to and collect next example
        subseq_mid_idx += step_size
        subseq_last_idx += step_size

    return indicies