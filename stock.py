# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 11:13:06 2021

@author: ludex
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt

#loading train data
X_df = pd.read_csv('stock.csv')
X_values = X_df.iloc[:3000,-1].values
X_train = X_values.reshape(-1, 1)

#scalling the train data
scaler = MinMaxScaler(feature_range=(0,1))
X_train = scaler.fit_transform(X_train)

features_set = []
labels = []
for i in range(0, len(X_train)-20):
    features_set.append(X_train[i:i+20])
    labels.append(X_train[i+20])


features_set, labels = np.array(features_set), np.array(labels)

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(features_set, labels, epochs = 5, batch_size = 40)

#loading test data

Y_df = pd.read_csv('stock.csv')
Y_values = Y_df.iloc[3001:,-1].values
Y_test = Y_values.reshape(-1, 1)

Y_test = scaler.transform(Y_test)


test_features = []
true_labels = []
for i in range(0, len(Y_test)-20):
    test_features.append(Y_test[i:i+20])
    true_labels.append(Y_test[i+20])
    
test_features = np.array(test_features)
test_features = np.reshape(test_features, (test_features.shape[0], 20, 1))

predictions = model.predict(test_features)
predictions_unscaled = scaler.inverse_transform(predictions)
true_labels_unscaled = scaler.inverse_transform(true_labels)

#evaluations
plt.figure(figsize=(10,6))
plt.plot(true_labels_unscaled, color='blue', label='Actual Stock Price')
plt.plot(predictions_unscaled , color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()