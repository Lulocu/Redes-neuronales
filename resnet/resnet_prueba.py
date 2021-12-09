from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import argparse
import tensorflow as tf
import time

import utils


parser = argparse.ArgumentParser(description='Trains a convolutional network for traffic prediction.')
files_group = parser.add_argument_group('Data files')
files_group.add_argument('-d', '--datasets', type=str, help='list of files to make training and validation sets',
                         nargs='+', metavar=('FILE 1', 'FILE 2'))
files_group.add_argument('-v', '--valid_set', type=str, help='list of validation set files', nargs='+',
                         metavar=('FILE 1', 'FILE 2'))
files_group.add_argument('-t', '--test_set', type=str, help='file of the test set data', nargs='+',
                         metavar=('FILE 1', 'FILE 2'))
prediction_group = parser.add_argument_group('Prediction parameters')
prediction_group.add_argument('-tw', '--time_window', default=12, type=int, help='time window used to predict')
prediction_group.add_argument('-ta', '--time_aggregation', default=1, type=int, help='steps aggregated for net input')
prediction_group.add_argument('-fw', '--forecast_window', default=1, type=int, help='time window to be predicted')
prediction_group.add_argument('-fa', '--forecast_aggregation', default=1, type=int, help='steps aggregated in forecast')
training_group = parser.add_argument_group('Training parameters')
training_group.add_argument('-ts', '--train_set_size', default=70000, type=int, help='training set size')
training_group.add_argument('-vs', '--valid_set_size', default=30000, type=int, help='validation set size')
training_group.add_argument('-vp', '--valid_partitions', default=100, type=int, help='validation set partitions number')
training_group.add_argument('-tp', '--test_partitions', default=100, type=int, help='test set partitions number')
training_group.add_argument('-b', '--batch_size', default=70, type=int, help='batch size for SGD')
training_group.add_argument('-l', '--learning_rate', default=1e-4, type=float, help='learning rate for SGD')
training_group.add_argument('-dr', '--decay_rate', default=0.1, type=float, help='learning rate decay rate')
training_group.add_argument('-ds', '--decay-steps', default=1000, type=int, help='learning rate decay steps')
training_group.add_argument('-c', '--gradient_clip', default=40.0, type=float, help='clip at this max norm of gradient')
training_group.add_argument('-m', '--max_steps', default=10000, type=int, help='max number of iterations for training')
training_group.add_argument('-s', '--save', action='store_true', help='save the model every epoch')
training_group.add_argument('-ens', '--ensemble', default=1, type=int, help='Number of the model in the ensemble')
args = parser.parse_args()

pickle_filename = utils.get_dataset_name(args.time_window, args.time_aggregation, args.forecast_window,
                                         args.forecast_aggregation, args.train_set_size, args.valid_set_size)

dataset = utils.get_dataset(pickle_filename, args, parser)
train_set = dataset[0]
train_labels = dataset[1]
valid_set = dataset[2]
valid_labels = dataset[3]
valid_set2 = dataset[4]
valid_labels2 = dataset[5]
test_set = dataset[6]
test_labels = dataset[7]
mean = dataset[8]
stddev = dataset[9]
del dataset

print('Training set', train_set.shape, train_labels.shape)
print('Validation set', valid_set.shape, valid_labels.shape)
print('Test set', valid_set2.shape, valid_labels2.shape)

print('Building model...')




net = (train_set - mean) / stddev  # Normalization of input

net = tf.pad(net, [[0, 0], [0,0], [0, 0], [14, 15]], "CONSTANT")
input_res1=keras.Input(shape = (27,12,3))
x = layers.Conv2D(32,[3,3],[1,1],activation=None,padding="same")(input_res1) #Tal vez falta el scope,scope=scope + '_1'
x = layers.ReLU()(x)
x = layers.BatchNormalization()(x)#No parece muy igual
x = layers.Conv2D(32,[3,3],[1,1],activation=None,padding="same")(x)
x = layers.BatchNormalization()(x)
xd=x+net
output_res1 = tf.nn.relu(xd)





model = keras.Model(inputs=input_res1, outputs=output_res1, name="res_model")

'''
Comprobaciones del modelo en consola y como png

'''

model.summary()

keras.utils.plot_model(model, "res_model.png", show_shapes=True)

'''
Compilación modelo

'''
print('\nComienza la compilación \n')

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

#Entrenamiento
history = model.fit(train_set, train_labels, batch_size=64, epochs=2, validation_split=0.2)

#Evaluación
test_scores = model.evaluate(valid_set, valid_labels, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
