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
file_dataset = '2015Flow.csv'
#file_test = '2016Prueba.csv'
#file_validate = '2016Prueba.csv'
filename_dataset = os.path.join(path,file_dataset)

data_file = pd.read_csv(filename_dataset)


column_indices = {name: i for i, name in enumerate(data_file.columns)}

n = len(data_file)
train_df = data_file[0:int(n*0.7)]
val_df = data_file[int(n*0.7):int(n*0.9)]
test_df = data_file[int(n*0.9):]

num_features = data_file.shape[1]


#Normalización
train_mean = train_df.mean()
train_std = train_df.std()
train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std


w2 = model.WindowGenerator(input_width=6, label_width=1, shift=1,train_df=train_df, val_df=val_df, test_df=test_df,
                     label_columns=['flow'])



model.WindowGenerator.split_window = model.split_window


model.WindowGenerator.plot = utils.plot



model.WindowGenerator.make_dataset = utils.make_dataset


model.WindowGenerator.train = utils.train
model.WindowGenerator.val = utils.val
model.WindowGenerator.test = utils.test
model.WindowGenerator.example = utils.example

# Each element is an (inputs, label) pair.
w2.train.element_spec

for example_inputs, example_labels in w2.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')

  single_step_window = model.WindowGenerator(
    input_width=1, label_width=1, shift=1,train_df=train_df, val_df=val_df, test_df=test_df,
    label_columns=['flow'])
single_step_window

class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]

baseline = Baseline(label_index=column_indices['flow'])

baseline.compile(loss=tf.losses.MeanSquaredError(),
                 metrics=[tf.metrics.MeanAbsoluteError()])

val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)

wide_window = model.WindowGenerator(
    input_width=24, label_width=24, shift=1,train_df=train_df, val_df=val_df, test_df=test_df,
    label_columns=['flow'])

print(wide_window)

print('Input shape:', wide_window.example[0].shape)
print('Output shape:', baseline(wide_window.example[0]).shape)

wide_window.plot(plot_col='flow',model=baseline)

MAX_EPOCHS = 20

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history


CONV_WIDTH = 3
conv_window = model.WindowGenerator(
    input_width=CONV_WIDTH,
    label_width=1,
    shift=1,
    train_df=train_df, val_df=val_df, test_df=test_df,
    label_columns=['flow'])

LABEL_WIDTH = 24
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = model.WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1,
    train_df=train_df, val_df=val_df, test_df=test_df,
    label_columns=['flow'])
conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(CONV_WIDTH,),
                           activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
])

print("Conv model on `conv_window`")
print('Input shape:', conv_window.example[0].shape)
print('Output shape:', conv_model(conv_window.example[0]).shape)


history = compile_and_fit(conv_model, conv_window)

IPython.display.clear_output()
val_performance['Conv'] = conv_model.evaluate(conv_window.val)
performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)

wide_conv_window.plot(plot_col='flow',model=conv_model)

