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
from keras.layers import Dense, SimpleRNN, BatchNormalization
import numpy as np



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
    print('File not found, try to run previous crowdsignals scripts first!')
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

train_X, test_X, train_y, test_y = prepare.split_multiple_datasets_classification([dataset1, dataset2, dataset3], ['label'], 'like', 0.7, filter=True, temporal=True, unknown_users=True)

train_y_no_dummy = train_y
test_y_no_dummy = test_y

train_y = pd.get_dummies(train_y)
test_y = pd.get_dummies(test_y)


train_y_mode = train_y_no_dummy.iloc[::10, :].mode(axis=1)[0]
train_y_mode = pd.get_dummies(train_y_no_dummy)

test_y_mode = test_y_no_dummy.iloc[::10, :].mode(axis=1)[0]
test_y_mode = pd.get_dummies(test_y_no_dummy)


from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, BatchNormalization
import numpy as np

num_classes = train_y.shape[1]
batch_size = 16
feature_dim = train_X.shape[1]  # Adjusted to use the correct dimension

# # Reshape your input data to have the appropriate shape
# train_X = np.random.random([965, feature_dim]).astype(np.float32)


# train_y = train_y_mode
# test_y = test_y_mode


train_X = np.reshape(train_X.values[:900], (-1, 10, train_X.shape[1]))  # Reshape to (samples, timesteps, features)
train_y = np.reshape(train_y_mode.values[:90], (90, 7))  # Reshape to (samples, timesteps, features)
# print(train_X)

# test_X = np.random.random([415, feature_dim]).astype(np.float32)
test_X = np.reshape(test_X.values[:270], (-1, 10, test_X.shape[1]))  # Reshape to (samples, timesteps, features)
test_y = np.reshape(test_y_mode.values[:27], (27,7))  # Reshape to (samples, timesteps, features)
# print(test_X)


# Define the model
model = Sequential()
model.add(SimpleRNN(num_classes, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])))  # RNN layer
model.add(Dense(num_classes, activation='softmax'))  # Output layer
model.add(BatchNormalization())


# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
print('train x shape',train_X.shape)
print('train y shape',train_y.shape)
print('test x shape',test_X.shape)
print('test y shape',test_y.shape)
model.fit(train_X, train_y, epochs=10, batch_size=batch_size)

# Evaluate the model
loss, accuracy = model.evaluate(test_X, test_y)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Make predictions
predictions = model.predict(test_X)
