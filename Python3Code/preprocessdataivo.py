import pandas as pd
import numpy as np
from pathlib import Path
import os

maps = ['IvoSquads', 'IvoLunges', 'IvoJumpingJacks', 'IvoLegRaises', 'IvoCrunches', 'IvoPushUps']


# Specify the directory containing the CSV files
directory = '/Users/florencecornelissen/Documents/VU/ML4QS/ML4QS2023A1/Python3Code/datasets/exercises/exercisesIvo'

# Create an empty dictionary to store the DataFrames
dataframes = {}

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv') and filename != 'device.csv':  # Check if the file is a CSV file
        file_path = os.path.join(directory, filename)
            
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)
            
        # Assign a name to the DataFrame (e.g., using the filename without the extension)
        name = filename[:-4]# Remove the '.csv' extension
        dataframes[name] = df
            
        # Process the DataFrame or perform desired operations
        # For example, you can print the contents of the DataFrame

maps = ['IvoLunges/', 'IvoJumpingJacks/', 'IvoLegRaises/', 'IvoCrunches/', 'IvoPushUps/']

# Specify the directory containing the CSV files
directory = '/Users/florencecornelissen/Documents/VU/ML4QS/ML4QS2023A1/Python3Code/datasets/exercises/exercisesIvo'

# Create an empty dictionary to store the DataFrames
additionaldfs = {}

for map in maps:
    path = os.path.join(directory, map)
    
    # Loop through all files in the directory
    for filename in os.listdir(path):
        if filename.endswith('.csv') and filename != 'device.csv':  # Check if the file is a CSV file
            file_path = os.path.join(path, filename)  # Use 'path' instead of 'directory'
            
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(file_path)
            
            # Assign a name to the DataFrame (e.g., using the filename without the extension)
            name = filename[:-4] + map[:-1] # Remove the '.csv' extension
            additionaldfs[name] = df
            
            # Process the DataFrame or perform desired operations
            # For example, you can print the contents of the DataFrame

# Concatenate all the dataframes of the same measurement type in one dataframe
files = []
for filename in os.listdir('./datasets/exercises/exercisesIvo/IvoCrunches/'):
    if filename.endswith('.csv'):
        files.append(filename[:-4])

combineddf = {}

for df in dataframes.keys():
    for df2 in additionaldfs.keys():
        if df in df2:
            combineddf[df] = pd.concat([dataframes[df], additionaldfs[df2]])
            dataframes[df] = combineddf[df]
            # print(combineddf[df])

for dataframe in dataframes:
    print(dataframe)


dataframes['Accelerometer'].rename(columns={'Time (s)': 'timestep', 'X (m/s^2)': 'x', 'Y (m/s^2)': 'y', 'Z (m/s^2)': 'z'}, inplace=True)
dataframes['Barometer'].rename(columns={'Time (s)': 'timestep', 'X (hPa)': 'x'}, inplace=True)
dataframes['Gyroscope'].rename(columns={'Time (s)': 'timestep', 'X (rad/s)': 'x', 'Y (rad/s)': 'y', 'Z (rad/s)': 'z'}, inplace=True)
dataframes['Linear_Accelerometer'].rename(columns={'Time (s)': 'timestep', 'X (m/s^2)': 'x', 'Y (m/s^2)': 'y', 'Z (m/s^2)': 'z'}, inplace=True)
dataframes['Location'].rename(columns={'Time (s)': 'timestep', 'Latitude (°)': 'latitude', 'Longitude (°)': 'longitude', 'Height (m)': 'h', 'Velocity (m/s)': 'vel', 'Direction (°)': 'dir', 'Horizontal Accuracy (m)': 'hor_acc', 'Vertical Accuracy (°)': 'ver_acc'}, inplace=True)
dataframes['Magnetometer'].rename(columns={'Time (s)': 'timestep', 'X (µT)': 'x', 'Y (µT)': 'y', 'Z (µT)': 'z'}, inplace=True)
dataframes['Proximity'].rename(columns={'Time (s)': 'timestep', 'Distance (cm)': 'dis'}, inplace=True)
dataframes['time'].rename(columns={'experiment time': 'experiment_time', 'system time': 'system_time', 'system time text': 'system_text_time'}, inplace=True)
dataframes['time'] = dataframes['time'].drop(columns=['system_time', 'system_text_time'], axis=1)

#cumsum = dataframes['time']['experiment_time'].cumsum()
# dataframes['timeseq'] = dataframes['time']['experiment_time'].cumsum()

print(dataframes['time'])
# Save to dataframes as csv in the datasets folder
for df in dataframes.keys():
    dataframes[df].to_csv('/Users/florencecornelissen/Documents/VU/ML4QS/ML4QS2023A1/Python3Code/datasets/exercises/DataIvo/' + df + 'ivo.csv', index=False)