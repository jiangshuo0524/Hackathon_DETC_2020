# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 11:19:52 2020

@author: mhrahman
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import keras
from keras import optimizers
from keras.models import Model
from keras.layers import Activation, Input,LSTM, GRU, Dropout, Dense

train_src = r'D:\Molla\Hackathon 2020\New folder\data for participants\Train_Organized\Bridgeport 1\Train'
csv_files = os.listdir(train_src)
os.chdir(train_src)

bridgeport1 = pd.read_csv('Train_X.csv',index_col = 0) 
bridgeport1 = bridgeport1.dropna()
Norm_data = normalize(bridgeport1)
X = Norm_data[:,1:-1]
Y = (Norm_data[:,-1]).reshape(-1,1)
X_train,X_test,Y_train, Y_test = train_test_split(X,Y)


def data_prep(csv):
    read_data = pd.read_csv(csv,index_col = 0)
    read_data = read_data.dropna()
    Norm_data = normalize(read_data)
    X = Norm_data[:,1:-1]
    Y = (Norm_data[:,-1]).reshape(-1,1)
    X_train,X_test,Y_train, Y_test = train_test_split(X,Y)
    return X_train,X_test,Y_train,Y_test

#Model
main_input = Input(shape=(None),name='main_input')
lstm_out = LSTM(units=64, activation= 'tanh')(main_input)
lstm_out = Dropout(0.2)(lstm_out)
lstm_out = Dense(32)(lstm_out)
lstm_out = Dense(1)(lstm_out)
main_output = Activation('softmax')(lstm_out)
model = Model(inputs = [main_input],outputs = main_output)
print(model.summary())

#Optimizer
lr = 0.001
epochs = 50

sgd=optimizers.SGD(lr=lr,momentum=0.9,nesterov=True)
adam=optimizers.Adam(lr=lr,beta_1=0.9,beta_2=0.999,epsilon=1e-08)
rmsprop=optimizers.rmsprop(lr=lr,rho=0.9,epsilon=None)
model.compile(optimizer=adam,loss='categorical_crossentropy')

#fitting
model.fit(X_train,Y_train,epochs=epochs,batch_size= 32)
test_output = model.predict(X_test)
