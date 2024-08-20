
import numpy as np 
import pandas as pd
import psutil
import multiprocessing
import os
import time
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from minio import Minio
from datetime import datetime
import config

labels = config.LABELS
secret_key = config.SECRET_KEY
access_key = config.ACCESS_KEY

def cpu_reader(
    pid: int, 
    return_dict
    ):
    cp = psutil.Process(pid=pid)
    cpu_reads = []
    while True:
        curr = cp.cpu_percent(interval=0.1)
        if curr > 0:
             cpu_reads.append(curr)     
        return_dict.put(cpu_reads)
        return_dict.get()

def data_grab(
    secret_key: str=secret_key,
    access_key: str=access_key
    ) -> pd.DataFrame:
    print('getting data...')
    client = Minio(
        endpoint="192.168.64.1:9000", 
        access_key=access_key, 
        secret_key=secret_key, 
        secure=False
        )

    obj_train = client.get_object(
        "bucket-test",
        "data/train.csv",
    )

    obj_test = client.get_object(
        "bucket-test",
        "data/test.csv",
    )

    df_train = pd.read_csv(obj_train)
    df_test = pd.read_csv(obj_test)
    df_merged = pd.concat([df_train, df_test])
    print('DONE')
    return df_merged

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
    print('\nMEASURING CPU USAGE')
    print(f'number of cores: {multiprocessing.cpu_count()}')
    pid = os.getpid()
    manager = multiprocessing.Manager()

    queue = manager.Queue()
    p = multiprocessing.Process(target=cpu_reader, args=(pid, queue))
    p.daemon = True
    p.start()
    time.sleep(2)

    df_merged = data_grab()

    final_q = queue.get()
    final_q = pd.Series(final_q)
    final_q = final_q[final_q > 0]

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

    queue2 = manager.Queue()
    p2 = multiprocessing.Process(target=cpu_reader, args=(pid, queue2))
    p2.daemon = True
    p2.start()
    time.sleep(2)

    rf_model = evaluate(rf_grid, X_train, y_train, X_test, y_test, labels)

    final_q2 = queue2.get()
    final_q2 = pd.Series(final_q2)
    final_q2 = final_q2[final_q2 > 0]

    label = f'local-{datetime.now()}'
    new_row = {
        'label':label,
        'data_cpu_usage':final_q.mean(),
        'ml_cpu_usage':final_q2.mean()
        }
    df_to_save = pd.read_csv('../results/results_cpu.csv')
    df_to_save = df_to_save.append(new_row, ignore_index=True)
    df_to_save
    df_to_save.to_csv('../results/results_cpu.csv', index=False)


