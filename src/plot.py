from typing import Optional, List
from datetime import timedelta

import pandas as pd
import plotly.express as px 

def plot_one_sample(
    example_id: int,
    features: pd.DataFrame,
    targets: Optional[pd.Series] = None,
    predictions: Optional[pd.Series] = None,
    display_title: Optional[bool] = True,
):
    """"""
    features_ = features.iloc[example_id]
    
    if targets is not None:
        target_ = targets.iloc[example_id]
    else:
        target_ = None
    
    ts_columns = [c for c in features.columns if c.startswith('rides_previous_')]
    ts_values = [features_[c] for c in ts_columns] + [target_]
    ts_dates = pd.date_range(
        features_['pickup_hour'] - timedelta(hours=len(ts_columns)),
        features_['pickup_hour'],
        freq='H'
    )
    
    # line plot with past values
    title = f'Pick up hour={features_["pickup_hour"]}, location_id={features_["pickup_location_id"]}' if display_title else None
    fig = px.line(
        x=ts_dates, y=ts_values,
        template='plotly_dark',
        markers=True, title=title
    )
    
    if targets is not None:
        # green dot for the value we wanna predict
        fig.add_scatter(x=ts_dates[-1:], y=[target_],
                        line_color='green',
                        mode='markers', marker_size=10, name='actual value') 
        
    if predictions is not None:
        # big red X for the predicted value, if passed
        prediction_ = predictions.iloc[example_id]
        fig.add_scatter(x=ts_dates[-1:], y=[prediction_],
                        line_color='red',
                        mode='markers', marker_symbol='x', marker_size=15,
                        name='prediction')             
    return fig


def plot_ts(
    ts_data: pd.DataFrame,
    locations: Optional[List[int]] = None
    ):
    """
    Plot time-series data
    """
    ts_data_to_plot = ts_data[ts_data.pickup_location_id.isin(locations)] if locations else ts_data

    fig = px.line(
        ts_data,
        x="pickup_hour",
        y="rides",
        color='pickup_location_id',
        template='none',
    )

    fig.show()


# change col-names??



import matplotlib.pyplot as plt

def hello(example_id: int, features: pd.DataFrame, targets: pd.Series):


    # Extract the feature and target for the given example_id
    feature_row = features.iloc[example_id]
    target_value = targets.iloc[example_id]
    
    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot features
    plt.plot(range(len(feature_row) - 2), feature_row[:-2], label='Features')
    
    # Marking the target value
    plt.axhline(y=target_value, color='r', linestyle='--', label='Target')

    # Adding labels and title
    plt.xlabel('Hour')
    plt.ylabel('Number of Rides')
    plt.title(f'Example {example_id} - Target: {target_value}')
    plt.legend()
    
    plt.show()


