# Imports
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from minio import Minio
from datetime import datetime
import config

# initialize global variables
secret_key = config.SECRET_KEY
access_key = config.ACCESS_KEY
labels = config.LABELS
bucket_name = config.BUCKET_NAME

# function which pulls the data from the designated data source, and saves it as a single dataframe.
@profile # memory_profiler decorator
def data_grab(
    secret_key: str = secret_key, 
    access_key: str = access_key,
    bucket_name: str = bucket_name
    ) -> pd.DataFrame:
    print('getting data...')
    # Initialize minio client, authorised with valid credentials
    client = Minio(
        endpoint="192.168.64.1:9000", 
        access_key=access_key, 
        secret_key=secret_key, 
        secure=False
        )
    # Get train data
    obj_train = client.get_object(
        bucket_name,
        "data/train.csv",
    )
    # Get test data
    obj_test = client.get_object(
        bucket_name,
        "data/test.csv",
    )
    # Read downloaded csv objects as dataframes
    df_train = pd.read_csv(obj_train)
    df_test = pd.read_csv(obj_test)
    df_merged = pd.concat([df_train, df_test])
    print('DONE')
    # Return merged dataframe
    return df_merged

# Function which fits and trains the model and predicts the values
@profile
def evaluate(
    model: GridSearchCV, 
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_test: pd.DataFrame, 
    labels: list
    ) -> GridSearchCV:
    print('model fit&predict...')
    # Fit model with train data
    model.fit(X_train, y_train)  
    # Predict values for test data
    pred = model.predict(X_test)
    print('DONE')
    # Communicate best set of parameters found during grid search
    print('Best Parameters:{}'.format(model.best_params_))
    # Communicate best test accuracy across cross validation folds
    print('Best Cross Validation score:{}'.format(model.best_score_))
    # Return trained model
    return model
   
if __name__ == '__main__':
    print('\nMEASURING TIME ELAPSED AND MEMORY USAGE')
    # Pull the data
    df_merged = data_grab()
    # Drop indexes and labels
    X = pd.DataFrame(df_merged.drop(['Activity','subject'],axis=1))
    # Save labels
    y = df_merged.Activity.values.astype(object) 
    # Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100)

    # Initialize Random Forest object
    rf = RandomForestClassifier()
    # Define parameters' grid for Random Forest - this will be explored during Grid Search
    param_grid = {
    #     'max_depth': [7],
    #     'criterion' : ['entropy'],
    #     'max_features': [25],
    #     'min_weight_fraction_leaf': [0.15],
    #     'n_estimators': [300]
        'random_state': [1,2,3,4,5,6,7,8,9]
    }
    # Initialize GridSeachCV object with Random Forest Classifier and defined parameters' grid
    # n_jobs=-1 ensures that all available cores will be utilized in this process
    rf_grid = GridSearchCV(rf, param_grid, n_jobs=-1)
    # Fit, train the model, predict values and communicate results
    rf_model = evaluate(rf_grid, X_train, y_train, X_test, y_test, labels)
