from datetime import datetime
from typing import Tuple
import pandas as pd

# splits tabular-dataset given dataframe, the cutoff the split into train/test, the name of the target column
def train_test_split(df, cutoff_date, target_column_name):

    # the training-data including feautres/targets is equal to the dataset where all the rows whose pickup-hour column is less than curoff date
    train_data = df[df.pickup_hour < cutoff_date].reset_index(drop=True)
    # the testing-data including features/targets is equal to the dataset where all the rows whose pickup-hour column is greater than cutoff-date
    test_data = df[df.pickup_hour >= cutoff_date].reset_index(drop=True)

    # the feature-train-data is the training-data but dropping the target-column passing in the target-col-name
    X_train = train_data.drop(columns=[target_column_name])
    # the target-train-data is the training-data but only selecting the target-column
    y_train = train_data[target_column_name]
    # doing same for testing-data
    X_test = test_data.drop(columns=[target_column_name])
    y_test = test_data[target_column_name]

    return X_train, y_train, X_test, y_test