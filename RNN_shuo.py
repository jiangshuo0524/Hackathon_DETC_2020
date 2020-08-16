#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 15:17:13 2020

@author: ubuntu
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import layers
from keras import optimizers
from keras.models import Model
from keras.layers import Activation, Input,LSTM, GRU, Dropout, Dense
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

data_X = pd.read_csv("./processed_data/Drill/Train_X.csv")
data_X = data_X.dropna().values
data_Y = pd.read_csv("./processed_data/Drill/Train_Y.csv")
data_Y = data_Y.dropna().values
data_X_X = data_X[:,2:7]
data_X_Y = data_X[:,7]
data_Y_X = data_Y[:,2:7]
data_Y_Y = data_Y[:,7]

X_all = np.hstack([data_X_X,data_Y_X])
y_axis1 = data_X_Y
y_axis2 = data_Y_Y
#print(X_all.shape)
#print(y_axis1.shape)
#print(y_axis2.shape)

DATA = X_all
LABEL = y_axis1

X_train, X_val, y_train, y_val = train_test_split(DATA, LABEL, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

X_train = np.array(X_train, dtype='float')
X_val = np.array(X_val, dtype='float')
X_test = np.array(X_test, dtype='float')
y_train = np.array(y_train, dtype='float')
y_val = np.array(y_val, dtype='float')
y_test = np.array(y_test, dtype='float')

print(X_train.shape, X_val.shape, X_test.shape)

def build_model():
    
    input_layer = Input(shape=[None, DATA.shape[1]])
    lstm_layer = LSTM(128)(input_layer)
    drop_layer = Dropout(0.5)(lstm_layer)
    dense_layer = Dense(128,activation='relu')(drop_layer)
    dense_layer = Dense(128,activation='relu')(drop_layer)
    output_layer = Dense(1,activation='sigmoid')(dense_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    optimizer = tf.keras.optimizers.RMSprop(0.00001)
    
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mse'])
    return model

model = build_model()
model.summary()

#fitting
history = model.fit(X_train,y_train,epochs=5, batch_size= 2,
                    validation_data=(X_val, y_val))

plt.rcParams['figure.dpi'] = 100

plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('MSE')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

p_train = model.predict(X_train)
p_val = model.predict(X_val)
p_test = model.predict(X_test)

print("R^2_train:",r2_score(y_train, p_train))
print("R^2_val:",r2_score(y_val, p_val))
print("R^2_test:",r2_score(y_test, p_test))

data_X = pd.read_csv("./processed_data/Drill/Test_X.csv")
data_X = data_X.drop(columns=["Unnamed: 7"])
data_X = data_X.dropna().values
data_X_X = data_X[:,2:7]
X_all = np.reshape(data_X_X, (data_X_X.shape[0], 1, data_X_X.shape[1]))

time_step = data_X[:,0].squeeze()
final_prediction = model.predict(X_all).squeeze()
df_final = pd.DataFrame({"time": time_step, "damage": final_prediction})
df_final.to_csv('./results/X_4.csv')



from matplotlib import pyplot as plt
plt.rcParams['figure.dpi'] = 200

plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('MSE')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()