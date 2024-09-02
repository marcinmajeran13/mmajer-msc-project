import pandas as pd
from datetime import datetime 

# Read mprofile.dat file containing time nad memory measurements
df = pd.read_csv('mprofile.dat', delim_whitespace=True, skiprows=1, names=['1','2','3','4','5','6','7'])
# Target summary rows from mprofile.dat
df_ts = df[df['1']=='FUNC']
# Extract data_grab() duration time, [end_ts] - [start_ts]
data_ts = df_ts['6'].iloc[0] - df_ts['4'].iloc[0]
# Extract evaluate() duration time, [end_ts] - [start_ts]
ml_ts = df_ts['6'].iloc[1] - df_ts['4'].iloc[1]
# Extract used memory
data_memory = df_ts['5'].iloc[0]
ml_memory = df_ts['5'].iloc[1]
# Extract memory increments
data_memory_inc = df_ts['5'].iloc[0] - df_ts['3'].iloc[0]
ml_memory_inc = df_ts['5'].iloc[1] - df_ts['3'].iloc[1]

# Create a new row that will be inserted into results file
label = f'local-{datetime.now()}'
new_row = {
    'label':label,
    'data_time':data_ts,
    'data_memory':data_memory,
    'data_memory_inc':data_memory_inc,
    'ml_time':ml_ts,
    'ml_memory':ml_memory,
    'ml_memory_inc':ml_memory_inc
    }
# Get results file, add new row and save it
df_to_save = pd.read_csv('../results/results.csv')
df_to_save = df_to_save.append(new_row, ignore_index=True)
df_to_save
df_to_save.to_csv('../results/results.csv', index=False)
