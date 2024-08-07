#!/usr/bin/env python
# coding: utf-8
import numpy as np 
import pandas as pd
import psutil
import multiprocessing
import os
import time
from io import StringIO
from google.cloud import storage
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from google.cloud import storage
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from datetime import timedelta, datetime

def cpu_reader(pid, return_dict):
    print(f'number of cores: {multiprocessing.cpu_count()}')
    cp = psutil.Process(pid=pid)
    cpu_reads = []
    while True:
        curr = cp.cpu_percent(interval=0.1)
        if curr > 0:
             cpu_reads.append(curr)     
        return_dict.put(cpu_reads)
        return_dict.get()

def data_grab(bucket_name):
    print('getting data...')
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    obj_train = bucket.get_blob('train.csv')
    obj_test = bucket.blob('test.csv')

    train_str = obj_train.download_as_text()
    test_str = obj_test.download_as_text()

    df_train = pd.read_csv(StringIO(train_str))
    df_test = pd.read_csv(StringIO(test_str))

    df_merged = pd.concat([df_train, df_test])
    print('DONE')
    return df_merged

def evaluate(model, X_train, y_train, X_test, y_test, labels):
    print('model fit&predict...')
    model.fit(X_train, y_train)  
    pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=pred)
    print('DONE')
    print('Best Parameters:{}'.format(model.best_params_))
    print('Best Cross Validation score:{}'.format(model.best_score_))
    return model

if __name__ == '__main__':
    print('\nMEASURING CPU USAGE')
    pid = os.getpid()
    manager = multiprocessing.Manager()

    queue = manager.Queue()
    p = multiprocessing.Process(target=cpu_reader, args=(pid, queue))
    p.daemon = True
    p.start()
    time.sleep(2)

    df_merged = data_grab('test_bucket_mmajer')

    final_q = queue.get()
    final_q = pd.Series(final_q)
    final_q = final_q[final_q > 0]

    X = pd.DataFrame(df_merged.drop(['Activity','subject'],axis=1))
    y = df_merged.Activity.values.astype(object) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100)
    labels = ['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS',
           'WALKING_UPSTAIRS']   

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

    label = f'cloud-{datetime.now() + timedelta(hours=2)}'
    new_row = {
        'label':label,
        'data_cpu_usage':final_q.mean(),
        'ml_cpu_usage':final_q2.mean()
        }
    client = storage.Client()
    bucket = client.get_bucket('test_bucket_mmajer')

    blob = bucket.blob('results/results_cpu_containerized.csv')
    blob.download_to_filename('results_cpu_containerized.csv')

    df_to_save = pd.read_csv('results_cpu_containerized.csv')
    df_to_save = df_to_save._append(new_row, ignore_index=True)
    df_to_save
    df_to_save.to_csv('results_cpu_containerized.csv', index=False)

    blob = bucket.blob(f'results/results_cpu_containerized.csv')
    blob.upload_from_filename('results_cpu_containerized.csv')

