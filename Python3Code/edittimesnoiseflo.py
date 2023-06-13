import pandas as pd
import numpy as np
from pathlib import Path
import os

maps = ['FloSquads/', 'FloLunges/', 'FloJumpingJacks/', 'FloLegRaises/', 'FloCrunches/', 'FloPushUps/']

# Specify the directory containing the CSV files
directory = '/Users/florencecornelissen/Documents/VU/ML4QS/ML4QS2023A1/Python3Code/datasets/exercises/exercisesFlo/'

# Create an empty dictionary to store the DataFrames
dfs = {}
dfsall = {}

for map in maps:
    path = os.path.join(directory, map)
    
    # Loop through all files in the directory
    for filename in os.listdir(path):
        if filename == 'Accelerometer.csv':  # Check if the file is a CSV file
            file_path = os.path.join(path, filename)  # Use 'path' instead of 'directory'
            
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(file_path)
            
            # Assign a name to the DataFrame (e.g., using the filename without the extension)
            name = filename[:-4] + map[:-1] # Remove the '.csv' extension
            dfsall[name] = df   

timesdf = pd.read_csv('/Users/florencecornelissen/Documents/VU/ML4QS/ML4QS2023A1/Python3Code/datasets/exercises/DataFlo/timeflo.csv')

indicesbegin = [234, 193, 115, 186, 199, 360]
begintimes = []
for df1 in dfsall.values():
    if indicesbegin == []:
        break
    begintimes.append(df1.at[indicesbegin[0], 'Time (s)'])
    indicesbegin.pop(0)

indicesend = [2540, 3841, 3136, 2647, 3961, 1975]
endtimes = []
for df1 in dfsall.values():
    if indicesend == []:
        break
    endtimes.append(df1.at[indicesend[0], 'Time (s)'])
    indicesend.pop(0)

timesdf['experiment_time'] = timesdf['experiment_time'].cumsum()

new_rows = []

for i, row in timesdf.iterrows():
    if row['event'] == 'START':
        new_rows.append({'event': 'start_timing', 'experiment_time': row['experiment_time']})
    new_rows.append(row)  # Add the current row
    if row['event'] == 'PAUSE':
        new_rows.append({'event': 'end_timing', 'experiment_time': row['experiment_time']})

# Create a new DataFrame with the updated rows
timesdf = pd.DataFrame(new_rows, columns=['event', 'experiment_time'])

# Reset the index
timesdf.reset_index(drop=True, inplace=True)

labels = ['squad', 'lunge', 'jumpingjack', 'legraise', 'crunch', 'pushup']

i = 0
for i in timesdf.index:
    if begintimes == [] or endtimes ==[] or labels == []:
        break
    if timesdf.loc[i]['event'] == 'start_timing':
        timesdf.at[i, 'label'] = 'switch'
    if timesdf.loc[i]['event'] == 'end_timing':
        timesdf.at[i, 'label'] = 'switch'
    if timesdf.loc[i]['event'] == 'START':
        # timesdf.loc[i-0.5] = ['start_timing', 0]
        timesdf.loc[i, 'experiment_time'] = begintimes[0] + timesdf.loc[i-1, 'experiment_time']
        timesdf.loc[i, 'event'] = 'start_exercise'
        timesdf.at[i, 'label'] = labels[0]
        # timesdf.at[i-0.5, 'label'] = 'switch'
        begintimes.pop(0)
    if timesdf.loc[i]['event'] == 'PAUSE':
        # timesdf.loc[i+0.5] = ['end_timing', timesdf.loc[i]['experiment_time']]
        timesdf.loc[i, 'experiment_time'] = endtimes[0] + timesdf.loc[i-1, 'experiment_time']
        timesdf.loc[i, 'event'] = 'end_exercise'
        timesdf.at[i, 'label'] = labels[0]
        # timesdf.at[i+0.5, 'label'] = 'switch'
        endtimes.pop(0)
        labels.pop(0)

if endtimes:
    # timesdf.loc[i+1] = ['end_timing', timesdf.loc[i, 'experiment_time']]
    timesdf.loc[i, 'experiment_time'] = endtimes[0] + timesdf.loc[i-1, 'experiment_time']
    timesdf.loc[i, 'event'] = 'end_exercise'
    timesdf.at[i, 'label'] = labels[0]
    timesdf.at[i+1, 'label'] = 'switch'
    # timesdf.at[i+1, 'label'] = 'switch'

new_rows = []

for i, row in timesdf.iterrows():
    new_rows.append(row.to_dict())  # Add the current row

    if row['event'] == 'start_timing':
        next_row = timesdf.iloc[i + 1]  # Get the next row
        new_rows.append({'event': 'end_timing', 'experiment_time': next_row['experiment_time'], 'label': 'switch'})
    elif row['event'] == 'end_exercise':
        new_rows.append({'event': 'start_timing', 'experiment_time': row['experiment_time'], 'label': 'switch'})

# Create a new DataFrame with the updated rows
timesdf = pd.DataFrame(new_rows, columns=['event', 'experiment_time', 'label'])

# Reset the index
timesdf.reset_index(drop=True, inplace=True)

# Change the start_exercise and end_exercise to start_timing and end_timing
timesdf['event'] = timesdf['event'].replace(['start_exercise', 'end_exercise'], ['start_timing', 'end_timing'])

# Create a new DataFrame with the updated rows
newtimesdf = pd.DataFrame()
for i in (timesdf.index):
    if timesdf['event'][i] == 'start_timing':
        newtimesdf = newtimesdf.append({'label': timesdf.loc[i]['label'], 'start_timing': timesdf.loc[i]['experiment_time'], 'end_timing': timesdf.loc[i+1]['experiment_time']}, ignore_index=True)

newtimesdf = newtimesdf[['label', 'start_timing', 'end_timing']]

newtimesdf.to_csv('/Users/florencecornelissen/Documents/VU/ML4QS/ML4QS2023A1/Python3Code/datasets/exercises/DataFlo/timeflo.csv', index=True)