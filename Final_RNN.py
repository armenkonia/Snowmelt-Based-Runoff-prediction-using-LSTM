# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 09:45:09 2023

@author: armen
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 19:39:20 2022

@author: armen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import plotly.express as px # to plot the time series plot
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler,LabelEncoder,MinMaxScaler
import random as python_random
import os

#%%
import pickle
# open a file, where you stored the pickled data
file = open('C:/Users/armen/Desktop/Thesis/Python/Last work/Data/merged_data_modified.pk', 'rb')
# file = open('C:/Users/armen/Desktop/Thesis/Python/Last work/Data/merged_data.pk', 'rb')

data = pickle.load(file)
# data = pd.read_csv("C:/Users/armen/Desktop/Thesis/Data/All_combined.csv", index_col= 'Date')

#%%
# divide between input (X) and output (Y)
X_data = data.iloc[:,1:]
Y_data = pd.DataFrame(data.iloc[:,0])
# transform data into MinMax
scaler = MinMaxScaler(feature_range = (0,1))
X_data = scaler.fit_transform(data.iloc[:,1:])
Y_data = scaler.fit_transform(pd.DataFrame(data.iloc[:,0])) 
# divide between train test
split_train = 18*12 #around 77% test
split_test = len(data) - 18*12
X_train, X_test = X_data[:split_train,:], X_data[-split_test:,:]
Y_train, Y_test = Y_data[:split_train,:], Y_data[-split_test:,:]

train_data_dates = data.index[:split_train]
test_data_dates = data.index[-split_test:]

#%%

def df_to_X_Y_seperate_batches (X_data, Y_data, timestep, futurestep): #this works as long as timestep>futurestep
    X = []
    Y = []
    X_data_np = np.array(X_data)
    Y_data_np = np.array(Y_data)
    no_input = int(len(X_data)/timestep)
    for i in range(no_input-1):
        set1 = X_data_np[i*timestep:timestep+i*timestep]
        X.append(set1)
        # set2 = Y_data.iloc[timestep + i*timestep] 
        set2 = Y_data_np[timestep + i*timestep : timestep + i*timestep +futurestep]
        Y.append(set2)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y
def df_to_X_Y_consecutive (X_data, Y_data, timestep, futurestep): #this works as long as timestep>futurestep
    X = []
    Y = []
    X_data_np = np.array(X_data)
    Y_data_np = np.array(Y_data)
    no_input = len(X_data)-timestep
    for i in range(no_input-futurestep+1):
        set1 = X_data_np[i :timestep+i]
        X.append(set1)
        set2 = Y_data_np[timestep+i : timestep+i +(futurestep)]
        Y.append(set2)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

# X, Y = df_to_X_Y_consecutive (X_data, Y_data, 12, 3)
timestep = 3 
futurestep = 1
x_train, y_train = df_to_X_Y_consecutive (X_train, Y_train, timestep, futurestep)
x_test, y_test = df_to_X_Y_consecutive (X_test, Y_test, timestep, futurestep)

#%%
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(0)
python_random.seed(0)
tf.random.set_seed(0)
# create LSTM
model = Sequential()
model.add(InputLayer((timestep, np.shape(x_train)[2])))
# model.add(LSTM(20, activation="relu", return_sequences = True))
model.add(LSTM(16, activation="relu", return_sequences = True))
model.add(LSTM(16, activation="relu", dropout = 0)) 

# model.add(Dense(8, 'relu'))
model.add(Dense(futurestep, 'linear'))

model.summary()
model.compile(loss='mse', optimizer='adam', metrics=RootMeanSquaredError())

#%%

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
file_path = 'best_model.h5'
mc = ModelCheckpoint(file_path, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
r = model.fit(x_train, y_train, epochs=30, batch_size=1, validation_data=(x_test, y_test), callbacks=[es, mc],  verbose=1)
#%%
train_accuracy = model.evaluate(x_train, y_train, verbose=0)
test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print("test loss MSE, test acc RMSE:", np.round(test_accuracy, 4))
#%%
# plot losses and accuracy
fig = plt.figure(figsize=(20,7))
fig.add_subplot(121)

# Accuracy
plt.plot(r.epoch, r.history['root_mean_squared_error'], label = "rmse")
plt.plot(r.epoch, r.history['val_root_mean_squared_error'], label = "val_rmse")

plt.title("RMSE", fontsize=18)
plt.xlabel("Epochs", fontsize=15)
plt.ylabel("RMSE", fontsize=15)
plt.grid(alpha=0.3)
plt.legend()


#Adding Subplot 1 (For Loss)
fig.add_subplot(122)

plt.plot(r.epoch, r.history['loss'], label="loss")
plt.plot(r.epoch, r.history['val_loss'], label="val_loss")

plt.title("Loss", fontsize=18)
plt.xlabel("Epochs", fontsize=15)
plt.ylabel("Loss", fontsize=15)
plt.grid(alpha=0.3)
plt.legend()

plt.show()

#%%
#predictions
y_train_predict = model.predict(x_train).flatten()
y_test_predict = model.predict(x_test).flatten()

#%%

train_data_dates = data.index[timestep:split_train]
test_data_dates = data.index[-(split_test-timestep):]

#%%
# plot actual vs predictions
y_train_1D = y_train.flatten()
y_test_1D = y_test.flatten()

fig, axis = plt.subplots(2, figsize=(15, 5))
axis[0].plot(train_data_dates, y_train_1D, label="train")
axis[0].plot(train_data_dates, y_train_predict, label="train_pred")
axis[0].legend(loc='upper right')

axis[1].plot(test_data_dates, y_test_1D, label="test")
axis[1].plot(test_data_dates, y_test_predict, label="test_pred")
axis[1].legend(loc='upper right')

#%%
def nse(targets,predictions):
    return 1-(np.sum((targets-predictions)**2)/np.sum((targets-np.mean(predictions))**2))
print('Train R2 Score: ', r2_score(y_train_1D, y_train_predict))
print('Test R2 Score: ', r2_score(y_test_1D, y_test_predict))
print('MAE: ', mean_absolute_error(y_test_1D, y_test_predict))
print('RMSE: ',  sqrt(mean_squared_error(y_test_1D, y_test_predict)))
print('NSE: ', nse(y_train_1D, y_train_predict))

#%%
#next-day data correlation
Y_data_t1 = Y_data[1:]
Y_data_t0 = Y_data[:-1]

# plt.scatter (Y_data_t1, Y_data_t0)
print('R2 Score: ', r2_score(Y_data_t1, Y_data_t0))

#%%
# scaler = MinMaxScaler(feature_range = (0,1))
# scaler_data = scaler.fit_transform(data)
unscaled_data = scaler.inverse_transform(data)
inverse_y_test_predict = scaler.inverse_transform(y_test_predict)
