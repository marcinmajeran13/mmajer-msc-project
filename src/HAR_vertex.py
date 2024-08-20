import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
    df_train = pd.read_csv(f'../{bucket_name}/train.csv')
    df_test = pd.read_csv(f'../{bucket_name}/test.csv')

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
