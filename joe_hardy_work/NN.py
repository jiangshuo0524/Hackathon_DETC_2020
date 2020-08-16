import keras
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Input,Add,LSTM,SimpleRNN
from matplotlib import pyplot as plt
from joe_hardy_work.processed_data_load import X_train_total,X_test_total,y_train_total,y_test_total
from keras import backend as K
from sklearn.metrics import r2_score

def SS_res_(y_true,y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    return SS_res

def SS_tot_(y_true,y_pred):
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return SS_tot

def coeff_determination(y_true, y_pred):
    from keras import backend as K
    SS_res =  SS_res_(y_true,y_pred)
    SS_tot = SS_tot_(y_true,y_pred)
    return ( 1 - SS_res/(SS_tot) )



print(X_train_total.iloc[0,:],y_train_total.iloc[0])
# X_train_total = X_train_total.to_numpy().reshape((-1,1,6))
# X_test_total = X_test_total.to_numpy().reshape((-1,1,6))
# y_train_total = y_train_total.to_numpy().reshape((-1,1,1))
# y_test_total = y_test_total.to_numpy().reshape((-1,1,1))
print(X_train_total)

model = Sequential()
#model.add(LSTM(128))
model.add(Dense(64,input_dim=X_train_total.shape[1],activation='relu')) #X_train_total.shape[1]
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(
    loss='mse',
    optimizer='adam',
    metrics=[coeff_determination,SS_res_,SS_tot_]
)

print(model.summary())

history = model.fit(X_train_total,y_train_total,epochs=150,batch_size=32,validation_split=0.2)

#loss,accuracy = model.evaluate(X_test_total,y_test_total)
y_test_pred = model.predict(X_test_total)
print(r2_score(y_test_total,y_test_pred))


model.save('nn.model2')
print(loss,accuracy)

print(history.history)

plt.plot(history.history['loss'])
plt.show()

plt.plot(history.history['coeff_determination'][2:])
plt.show()


#keras.utils.plot_model(model,to_file='nn.png')

#input1 = Input(shape=X_train_total.shape[1],)


