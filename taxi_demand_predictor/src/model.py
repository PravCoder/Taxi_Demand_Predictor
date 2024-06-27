from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline 
import lightgbm as lgb  



# FEATURE ENGINEERING
def average_rides_last_4_weeks(X):  # given X-dataframe-train-data
    # create new column avr-rides-4-weeks = mean of the four columns of last 4 weeks
    X["average_rides_last_4_weeks"] = 0.25*(
        X[f"rides_previous_{7*24}_hour"] + \
        X[f"rides_previous_{2*7*24}_hour"] + \
        X[f"rides_previous_{3*7*24}_hour"] + \
        X[f"rides_previous_{4*7*24}_hour"] 
    )

    return X



# create custom Transformer-class that inherits from these two scikit learn classes
class TemporalFeaturesEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    # takes in X-df
    def transform(self, X=None):

        X_ = X.copy()  # create copy of X-df
        # perform feature-engineering, by creating new column-hour which is hour of the pickup-hour-column
        X_["hour"] = X_["pickup_hour"].dt.hour
        X_["day_of_week"] = X_["pickup_hour"].dt.dayofweek  # set new column day-of-week for each row equal to that rows pickup-hour-columns day part

        return X_.drop(columns=["pickup_hour"])  # drop this column
    
 
def get_pipeline(**hyperparams): # tive a set of hyperparameters passes those in model when creating pipeline

    # wraps the func-aver-rides-last-4-weeks so it can be used in scikit learn pipeline
    add_feature_average_rides_last_4_weeks = FunctionTransformer(average_rides_last_4_weeks, validate=False)
    
    # create custom-transformer-obj
    add_temporal_features = TemporalFeaturesEngineer()

    # creating pipeline passing sequence of transformations on X, and the model
    return make_pipeline(add_feature_average_rides_last_4_weeks, add_temporal_features, lgb.LGBMRegressor(**hyperparams))  # pass in given hyperparameters into model
