# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 12:25:05 2020

@author: mhrahman
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import os

train_src = r'D:\Molla\Hackathon 2020\New folder\data for participants\Train_Organized\Bridgeport 1\Train'
csv_files = os.listdir(train_src)
os.chdir(train_src)

bridgeport1 = pd.read_csv('Train_X.csv',index_col = 0) 
bridgeport1 = bridgeport1.dropna()

train_dataset_br = bridgeport1.sample(frac=0.8,random_state=0)
test_dataset_br = bridgeport1.drop(train_dataset_br.index)

train_stat_br = train_dataset_br.describe()
train_stat_br.pop('Damage Accumulation')
train_stat_br = train_stat_br.transpose()
train_stat_br

train_label_br = train_dataset_br.pop('Damage Accumulation')
test_label_br = test_dataset_br.pop('Damage Accumulation')

def norm(x):
  return (x - train_stat_br['mean']) / train_stat_br['std']
normed_train_data_br = norm(train_dataset_br)
normed_test_data_br = norm(test_dataset_br)

def build_model():
  model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=[len(train_dataset_br.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()
model.summary()

example_batch = normed_train_data_br[:10]
example_result = model.predict(example_batch)
example_result

EPOCHS = 100

history = model.fit( normed_train_data_br, train_label_br,epochs=EPOCHS)