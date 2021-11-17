import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import pandas as pd
import utils
import model
import IPython
import IPython.display


path = '/home/luis/Documentos/pruebaRedes/datasetsMini/'
file_dataset = '2015.csv'
file_test = '2016Prueba.csv'
#file_validate = '2016Prueba.csv'
filename_dataset = os.path.join(path,file_dataset)
filename_test = os.path.join(path,file_test)

data_file = pd.read_csv(filename_dataset)
test_file = pd.read_csv(filename_test)

column_indices = {name: i for i, name in enumerate(data_file.columns)}
column_indices_test = {name: i for i, name in enumerate(test_file.columns)}
n = len(data_file)
nt = len(test_file)
train_df = data_file
val_df = test_file[:int(nt*0.7)]
test_df = test_file[int(nt*0.7):]

#num_features = data_file.shape[1:]


#Normalización
train_mean = train_df.mean()
train_std = train_df.std()
train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std


#Añades al modelo los datasets necesarios
model.WindowGenerator.plot = utils.plot
model.WindowGenerator.make_dataset = utils.make_dataset
model.WindowGenerator.train = utils.train
model.WindowGenerator.val = utils.val
model.WindowGenerator.test = utils.test
model.WindowGenerator.example = utils.example

val_performance = {}
performance = {}


CONV_WIDTH = 3
conv_window = model.WindowGenerator(
    input_width=CONV_WIDTH,
    label_width=1,
    shift=1,
    train_df=train_df, val_df=val_df, test_df=test_df,
    label_columns=['flow','density','speed'])

LABEL_WIDTH = 24
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = model.WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1,
    train_df=train_df, val_df=val_df, test_df=test_df,

    label_columns=['flow','density','speed'])



conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(CONV_WIDTH,),
                           activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=3),
])
print("Conv model on `conv_window`")
print('Input shape:', conv_window.example[0].shape)
print('Output shape:', conv_model(conv_window.example[0]).shape)


history = model.compile_and_fit(conv_model, conv_window,max_epochs=1)

keras.utils.plot_model(conv_model, "CNNSEQ_model.png", show_shapes=True)


val_performance['Conv'] = conv_model.evaluate(conv_window.val)
performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)

wide_conv_window.plot(plot_col='flow',model=conv_model)
wide_conv_window.plot(plot_col='density',model=conv_model)
wide_conv_window.plot(plot_col='speed',model=conv_model)