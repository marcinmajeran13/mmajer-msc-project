from io import StringIO
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from google.cloud import storage
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import config

labels = config.LABELS
bucket_name = config.BUCKET_NAME

@profile
def data_grab(
    bucket_name: str=bucket_name
    ) -> pd.DataFrame:
    print('getting data...')
    # Initialize Cloud Storage Client
    # Instance from which the script is run needs to be already authorised
    # so program knows which GCP project to target
    client = storage.Client()
    # Get Cloud Storage bucket
    bucket = client.get_bucket(bucket_name)
    # Retrieve train and test files as blobs
    obj_train = bucket.get_blob('train.csv')
    obj_test = bucket.blob('test.csv')
    # Download train and test files as text
    train_str = obj_train.download_as_text()
    test_str = obj_test.download_as_text()
    # Cast to csv and read train and test files to dataframes
    df_train = pd.read_csv(StringIO(train_str))
    df_test = pd.read_csv(StringIO(test_str))

    df_merged = pd.concat([df_train, df_test])
    print('DONE')
    return df_merged
    
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
    model.fit(X_train, y_train)  
    pred = model.predict(X_test)
    print('DONE')
    print('Best Parameters:{}'.format(model.best_params_))
    print('Best Cross Validation score:{}'.format(model.best_score_))
    return model

if __name__ == '__main__':
    print('\nMEASURING TIME ELAPSED AND MEMORY USAGE')

    df_merged = data_grab()
    X = pd.DataFrame(df_merged.drop(['Activity','subject'],axis=1))
    y = df_merged.Activity.values.astype(object) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100)

    rf = RandomForestClassifier()
    param_grid = {
    #     'max_depth': [7],
    #     'criterion' : ['entropy'],
    #     'max_features': [25],
    #     'min_weight_fraction_leaf': [0.15],
    #     'n_estimators': [300]
        'random_state': [1,2,3,4,5,6,7,8,9]
    }
    rf_grid = GridSearchCV(rf, param_grid, n_jobs=-1)
    rf_model = evaluate(rf_grid, X_train, y_train, X_test, y_test, labels)
