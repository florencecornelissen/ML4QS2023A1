from util.VisualizeDataset import VisualizeDataset
from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Chapter7.Evaluation import RegressionEvaluation, ClassificationEvaluation
from Chapter8.LearningAlgorithmsTemporal import TemporalClassificationAlgorithms
from Chapter8.LearningAlgorithmsTemporal import TemporalRegressionAlgorithms
from Chapter7.FeatureSelection import FeatureSelectionClassification
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot
# from exercises_ch7_classification_individual import used_features

import sys
import copy
import pandas as pd
from util import util
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, SimpleRNN, BatchNormalization, Dropout
import numpy as np
from tensorflow import keras
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
import itertools


from pathlib import Path
import pandas as pd
from util.VisualizeDataset import VisualizeDataset

DATA_PATH = Path('./intermediate_datafiles/')
DATASET_FNAME1 = 'chapter5_resultIvo.csv'
DATASET_FNAME2 = 'chapter5_resultJoost.csv'
DATASET_FNAME3 = 'chapter5_resultFlo.csv'
# RESULT_FNAME =  'chapter3_result_outliers'+participant+'.csv'

# Next, import the data from the specified location and parse the date index.
try:
    dataset1 = pd.read_csv(DATA_PATH / DATASET_FNAME1, index_col=0)
    dataset2 = pd.read_csv(DATA_PATH / DATASET_FNAME2, index_col=0)
    dataset3 = pd.read_csv(DATA_PATH / DATASET_FNAME3, index_col=0)
except IOError as e:
    print('File not found, try to run previous exercises scripts first!')
    raise e

common_columns = set(dataset1.columns) & set(dataset2.columns) & set(dataset3.columns)
dataset1 = dataset1[common_columns]
dataset2 = dataset2[common_columns]
dataset3 = dataset3[common_columns]

dataset1.index = pd.to_datetime(dataset1.index)
dataset2.index = pd.to_datetime(dataset2.index)
dataset3.index = pd.to_datetime(dataset3.index)

# We'll create an instance of our visualization class to plot the results.
DataViz = VisualizeDataset()

# Of course we repeat some stuff from Chapter 3, namely to load the dataset

# Read the result from the previous chapter, and make sure the index is of the type datetime.


# Let us consider our second task, namely the prediction of the heart rate. We consider this as a temporal task.

prepare = PrepareDatasetForLearning()

selected_features= ['linacc_phone_y_freq_0.0_Hz_ws_50', 'acc_phone_y_freq_0.0_Hz_ws_50', 'acc_phone_y_freq_0.0_Hz_ws_50', 'gyr_phone_y_freq_0.0_Hz_ws_50', 'gyr_phone_y_freq_0.0_Hz_ws_50', 'mag_phone_x_max_freq', 'mag_phone_x_max_freq', 'acc_phone_z_freq_0.2_Hz_ws_50', 'acc_phone_z_freq_0.2_Hz_ws_50', 'gyr_phone_x_freq_0.2_Hz_ws_50', 'gyr_phone_x_freq_0.2_Hz_ws_50', 'acc_phone_y_freq_1.0_Hz_ws_50', 'acc_phone_y_freq_1.0_Hz_ws_50', 'linacc_phone_y_freq_0.2_Hz_ws_50', 'linacc_phone_y_freq_0.2_Hz_ws_50', 'linacc_phone_z_freq_weighted', 'linacc_phone_z_freq_weighted', 'gyr_phone_x_freq_4.6_Hz_ws_50', 'gyr_phone_x_freq_4.6_Hz_ws_50', 'acc_phone_y_freq_0.6_Hz_ws_50', 'acc_phone_y_freq_0.6_Hz_ws_50', 'acc_phone_x_freq_1.0_Hz_ws_50', 'acc_phone_x_freq_1.0_Hz_ws_50', 'acc_phone_x_freq_1.2_Hz_ws_50', 'acc_phone_x_freq_1.2_Hz_ws_50', 'linacc_phone_x_max_freq', 'linacc_phone_x_max_freq', 'gyr_phone_x_freq_3.4_Hz_ws_50', 'gyr_phone_x_freq_3.4_Hz_ws_50', 'gyr_phone_x_freq_3.8_Hz_ws_50', 'gyr_phone_x_freq_3.8_Hz_ws_50', 'linacc_phone_z_freq_1.8_Hz_ws_50', 'linacc_phone_z_freq_1.8_Hz_ws_50', 'linacc_phone_x_freq_1.0_Hz_ws_50', 'linacc_phone_x_freq_1.0_Hz_ws_50', 'pca_7', 'pca_7', 'gyr_phone_x_freq_1.8_Hz_ws_50', 'gyr_phone_x_freq_1.8_Hz_ws_50', 'acc_phone_y_freq_4.4_Hz_ws_50', 'acc_phone_y_freq_4.4_Hz_ws_50', 'acc_phone_y_freq_2.0_Hz_ws_50', 'acc_phone_y_freq_2.0_Hz_ws_50', 'gyr_phone_x_freq_3.0_Hz_ws_50', 'gyr_phone_x_freq_3.0_Hz_ws_50', 'mag_phone_y_freq_0.6_Hz_ws_50', 'mag_phone_y_freq_0.6_Hz_ws_50', 'linacc_phone_x_freq_2.0_Hz_ws_50', 'linacc_phone_x_freq_2.0_Hz_ws_50', 'acc_phone_y_freq_2.4_Hz_ws_50', 'acc_phone_y_freq_2.4_Hz_ws_50', 'acc_phone_y_freq_2.6_Hz_ws_50', 'acc_phone_y_freq_2.6_Hz_ws_50', 'mag_phone_y_freq_0.4_Hz_ws_50', 'mag_phone_y_freq_0.4_Hz_ws_50', 'acc_phone_y_freq_1.4_Hz_ws_50', 'acc_phone_y_freq_1.4_Hz_ws_50', 'mag_phone_y_freq_2.4_Hz_ws_50', 'mag_phone_y_freq_2.4_Hz_ws_50', 'mag_phone_y_freq_3.0_Hz_ws_50', 'mag_phone_y_freq_3.0_Hz_ws_50', 'acc_phone_y', 'acc_phone_y', 'pca_6', 'pca_6', 'mag_phone_y_freq_4.0_Hz_ws_50', 'mag_phone_y_freq_4.0_Hz_ws_50', 'mag_phone_y_freq_0.8_Hz_ws_50', 'mag_phone_y_freq_0.8_Hz_ws_50', 'acc_phone_z_freq_4.2_Hz_ws_50']

train_X, test_X, train_y, test_y = prepare.split_multiple_datasets_classification([dataset1, dataset2, dataset3], ['label'], 'like', 0.7, filter=True, temporal=True, unknown_users=True)

train_y_no_dummy = train_y
test_y_no_dummy = test_y

train_X = train_X[selected_features]
test_X = test_X[selected_features]

train_y = pd.get_dummies(train_y)
test_y = pd.get_dummies(test_y)

train_y_mode = train_y_no_dummy.iloc[::10, :].mode(axis=1)[0]
train_y_mode = pd.get_dummies(train_y_no_dummy)

test_y_mode = test_y_no_dummy.iloc[::10, :].mode(axis=1)[0]
test_y_mode = pd.get_dummies(test_y_no_dummy)

num_classes = train_y.shape[1]
batch_size = 16
feature_dim = train_X.shape[1]  # Adjusted to use the correct dimension


train_X = np.reshape(train_X.values[:900], (-1, 10, train_X.shape[1]))  # Reshape to (samples, timesteps, features)
train_y = np.reshape(train_y_mode.values[:90], (90, 7))  # Reshape to (samples, timesteps, features)
# print(train_X)

# test_X = np.random.random([415, feature_dim]).astype(np.float32)
test_X = np.reshape(test_X.values[:270], (-1, 10, test_X.shape[1]))  # Reshape to (samples, timesteps, features)
test_y = np.reshape(test_y_mode.values[:27], (27,7))  # Reshape to (samples, timesteps, features)
# print(test_X)

np.random.seed(123)

# Normalize the input data
scaler = StandardScaler()
train_X = scaler.fit_transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

# Initialize the variables
num_iterations = 100
test_loss_values = []
test_accuracy_values = []

for i in range(num_iterations):
    # Rest of your code
    
    # Create a new model instance for each iteration
    model = Sequential()
    model.add(SimpleRNN(64, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(7, activation='softmax'))

    # Compile the model
    learning_rate = 0.0015
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Train the model
    history = model.fit(train_X, train_y, epochs=10, batch_size=batch_size, verbose=0)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(test_X, test_y, verbose=0)
    test_loss_values.append(loss)
    test_accuracy_values.append(accuracy)

# Calculate the means
mean_test_loss = np.mean(test_loss_values)
mean_test_accuracy = np.mean(test_accuracy_values)

# Print the means
print(f"Mean Test Loss: {mean_test_loss:.4f}")
print(f"Mean Test Accuracy: {mean_test_accuracy:.4f}")

# Plot the distribution of test loss values
plt.figure()
plt.hist(test_loss_values, bins=10, edgecolor='black')
plt.title('Distribution of Test Loss')
plt.xlabel('Test Loss')
plt.ylabel('Frequency')
plt.show()

# Plot the distribution of test accuracy values
plt.figure()
plt.hist(test_accuracy_values, bins=10, edgecolor='black')
plt.title('Distribution of Test Accuracy')
plt.xlabel('Test Accuracy')
plt.ylabel('Frequency')
plt.show()