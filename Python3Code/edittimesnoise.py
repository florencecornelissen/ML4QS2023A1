import pandas as pd
import numpy as np
from pathlib import Path
import os

maps = ['FloSquads/', 'FloLunges/', 'FloJumpingJacks/', 'FloLegRaises/', 'FloCrunches/', 'FloPushUps/']

# Specify the directory containing the CSV files
directory = './datasets/exercises/'

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

timesdf = pd.read_csv('./datasets/exercises/timeflo.csv')

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

labels = ['squad', 'lunge', 'jumpingjack', 'legraise', 'crunch', 'pushup']

i=0
for i in timesdf.index:
    if begintimes == []:
        break
    if (i%2 == 0):
        timesdf.loc[i, 'experiment_time'] = begintimes[0]
        begintimes.pop(0)
        timesdf.at[i, 'label'] = labels[0]
    else:
        timesdf.loc[i, 'experiment_time'] = endtimes[0]
        endtimes.pop(0)
        timesdf.at[i, 'label'] = labels[0]
        labels.pop(0)

if endtimes:
    timesdf.loc[i, 'experiment_time'] = endtimes[0]
    timesdf.at[i, 'label'] = labels[0]

# Create a pivot table indexed by label
pivottimes  = timesdf.pivot_table(index='label', columns='event', values='experiment_time')

pivottimes.to_csv('/Users/florencecornelissen/Documents/VU/ML4QS/ML4QS2023A1/Python3Code/datasets/exercises/timeflo.csv', index=True)