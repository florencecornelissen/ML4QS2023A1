import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime, timedelta

def preprocessdatatotal(participant, indicesbegin, indicesend):
    maps = [participant+'Squads', participant+'Lunges', participant+'JumpingJacks', participant+'LegRaises', participant+'Crunches', participant+'PushUps']


    # Specify the directory containing the CSV files
    directory = './datasets/exercises/exercises'+participant+'/'+participant+'Squads/'

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

    maps = [participant+'Lunges/', participant+'JumpingJacks/', participant+'LegRaises/', participant+'Crunches/', participant+'PushUps/']

    # Specify the directory containing the CSV files
    directory = './datasets/exercises/exercises'+participant+'/'

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
    for filename in os.listdir('./datasets/exercises/exercises'+participant+'/'+participant+'Crunches/'):
        if filename.endswith('.csv'):
            files.append(filename[:-4])

    combineddf = {}


    for df in dataframes.keys():
        for df2 in additionaldfs.keys():
            if df2.startswith(df):
                combineddf[df] = pd.concat([dataframes[df], additionaldfs[df2]])
                dataframes[df] = combineddf[df]
                # print(combineddf[df])

    dataframes['Accelerometer'].rename(columns={'Time (s)': 'timestep', 'X (m/s^2)': 'x', 'Y (m/s^2)': 'y', 'Z (m/s^2)': 'z'}, inplace=True)
    dataframes['Barometer'].rename(columns={'Time (s)': 'timestep', 'X (hPa)': 'x'}, inplace=True)
    dataframes['Gyroscope'].rename(columns={'Time (s)': 'timestep', 'X (rad/s)': 'x', 'Y (rad/s)': 'y', 'Z (rad/s)': 'z'}, inplace=True)
    dataframes['Linear_Accelerometer'].rename(columns={'Time (s)': 'timestep', 'X (m/s^2)': 'x', 'Y (m/s^2)': 'y', 'Z (m/s^2)': 'z'}, inplace=True)
    dataframes['Location'].rename(columns={'Time (s)': 'timestep', 'Latitude (°)': 'latitude', 'Longitude (°)': 'longitude', 'Height (m)': 'h', 'Velocity (m/s)': 'vel', 'Direction (°)': 'dir', 'Horizontal Accuracy (m)': 'hor_acc', 'Vertical Accuracy (°)': 'ver_acc'}, inplace=True)
    dataframes['Magnetometer'].rename(columns={'Time (s)': 'timestep', 'X (µT)': 'x', 'Y (µT)': 'y', 'Z (µT)': 'z'}, inplace=True)
    dataframes['Proximity'].rename(columns={'Time (s)': 'timestep', 'Distance (cm)': 'dis'}, inplace=True)
    dataframes['time'].rename(columns={'experiment time': 'experiment_time', 'system time': 'system_time', 'system time text': 'system_text_time'}, inplace=True)
    dataframes['time'] = dataframes['time'].drop(columns=['system_time', 'system_text_time'], axis=1)

    for dataframe in dataframes:
        if dataframe != 'time':
            temp = 0
            value =  0
            new_time = []
            for idx,row in dataframes[str(dataframe)].iterrows():
                dif = row['timestep'] - temp 
                if dif<0:
                    dif = 0
                value = value + dif
                new_time.append(value)
                prev_dif = dif
                temp = row['timestep']
            dataframes[str(dataframe)]['timestep'] = new_time

    # Save to dataframes as csv in the datasets folder
    for df in dataframes.keys():
        dataframes[df].to_csv('./datasets/exercises/Data'+participant+'/' + df + participant+ '.csv', index=False)

    maps = [participant+'Squads/', participant+'Lunges/', participant+'JumpingJacks/', participant+'LegRaises/', participant+'Crunches/', participant+'PushUps/']

    # Specify the directory containing the CSV files
    directory = './datasets/exercises/exercises'+participant+'/'

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

    timesdf = pd.read_csv('./datasets/exercises/Data'+participant+'/time'+participant+'.csv')

    begintimes = []
    for df1 in dfsall.values():
        if indicesbegin == []:
            break
        begintimes.append(df1.at[indicesbegin[0], 'Time (s)'])
        indicesbegin.pop(0)

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

    newtimesdf.to_csv('./datasets/exercises/Data'+participant+'/time'+participant+'.csv', index=False)

    # #### ADD THE TIMESTAMPS TO THE DATAFRAMES ####
    # maps = [participant+'Squads', participant+'Lunges', participant+'JumpingJacks', participant+'LegRaises', participant+'Crunches', participant+'PushUps']

    # # Specify the directory containing the CSV files
    # directory = './datasets/exercises/Data'+participant+'/'

    # # Create an empty dictionary to store the DataFrames
    # dataframes = {}

    # # Loop through all files in the directory
    # for filename in os.listdir(directory):
    #     if filename.endswith('.csv') and filename != 'device.csv':  # Check if the file is a CSV file
    #         file_path = os.path.join(directory, filename)
                
    #         # Read the CSV file into a pandas DataFrame
    #         df = pd.read_csv(file_path)
                
    #         # Assign a name to the DataFrame (e.g., using the filename without the extension)
    #         name = filename[:-4]# Remove the '.csv' extension
    #         dataframes[name] = df
                
    #         # Process the DataFrame or perform desired operations
    #         # For example, you can print the contents of the DataFrame

    # exercisedate = datetime(2023,6,7,14,13,30,636874)

    # for df in dataframes:
    #     if df != 'time'+participant:
    #         dataframes[df]['timestamps'] = exercisedate.timestamp() + dataframes[df]['timestep']
    #         dataframes[df].to_csv('./datasets/exercises/Data'+participant+'/'+df+'.csv', index=False)
    #     if df == 'time'+participant:
    #         dataframes[df]['label_start'] = exercisedate.timestamp() + dataframes[df]['start_timing']
    #         dataframes[df]['label_end'] = exercisedate.timestamp() + dataframes[df]['end_timing']
    #         dataframes[df].to_csv('./datasets/exercises/Data'+participant+'/'+df+'.csv', index=False)
