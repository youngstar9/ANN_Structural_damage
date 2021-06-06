#%%
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import losses,datasets,layers,optimizers,Sequential, metrics,models

gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)



import  os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
df = pd.read_excel(r'G:\Subject\小论文2\ml\train.xlsx')
cols = ['story','ld','drift','f','a','D']
df = df[cols]

#%%
train_dataset = df.sample(frac=1)
train_labels = train_dataset.pop('D')

train_stats = train_dataset.describe()
train_stats = train_stats.transpose()
# print(train_stats)



# 标准化数据
def norm(x):
    #return x
    return (x-train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
tf.keras.backend.clear_session()

model = models.Sequential()
model.add(layers.Dense(64,activation = tf.nn.swish,input_shape=(5,)))
#model.add(layers.Dense(512,activation = tf.nn.swish ))
model.add(layers.Dense(64,activation = tf.nn.swish ))
model.add(layers.Dense(64,activation = tf.nn.swish ))
#model.add(layers.Dense(32,activation = tf.nn.swish ))
#model.add(layers.Dense(64,activation = tf.nn.swish ))
model.add(layers.Dense(1))

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss='mse',metrics=['MAE'])
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(normed_train_data,train_labels,batch_size=64,epochs=50,
            validation_split=0.2,callbacks=[early_stop])

#%%

def plot_metric(history,metric):
    # sns.set_style("dark")
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1,len(train_metrics)+1)
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    # sns.set(font='SimHei')
    plt.plot(epochs,train_metrics,'b--')
    plt.plot(epochs,val_metrics,'r-')
    # plt.title('训练集和验证集的MSE')
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.legend(["训练集", '验证集'])
    plt.show()


plot_metric(history,"loss")
plot_metric(history,"MAE")







model.save(r'G:\Subject\小论文2\ml\sldfa\model',save_format="tf")
print('export saved model')

