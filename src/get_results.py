import pandas as pd
from datetime import timedelta, datetime 

df = pd.read_csv('mprofile.dat', sep='\s+', skiprows=1, names=['1','2','3','4','5','6','7'])
df_ts = df[df['1']=='FUNC']
data_ts = df_ts['6'].iloc[0] - df_ts['4'].iloc[0]
ml_ts = df_ts['6'].iloc[1] - df_ts['4'].iloc[1]
data_memory = df_ts['5'].iloc[0]
ml_memory = df_ts['5'].iloc[1]
data_memory_inc = df_ts['5'].iloc[0] - df_ts['3'].iloc[0]
ml_memory_inc = df_ts['5'].iloc[1] - df_ts['3'].iloc[1]
label = f'vertex-{datetime.now() + timedelta(hours=2)}-boost'
new_row = {
    'label':label,
    'data_time':data_ts,
    'data_memory':data_memory,
    'data_memory_inc':data_memory_inc,
    'ml_time':ml_ts,
    'ml_memory':ml_memory,
    'ml_memory_inc':ml_memory_inc
    }
df_to_save = pd.read_csv('../results/results_vertex.csv')
df_to_save = df_to_save._append(new_row, ignore_index=True)
df_to_save.to_csv('../results/results_vertex.csv', index=False)
