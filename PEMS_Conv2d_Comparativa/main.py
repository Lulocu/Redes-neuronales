from tensorflow.keras import backend
import argparse
import tensorflow as tf
import csv
import utils
import model
import os
import numpy as np
import matplotlib.pyplot as plt

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
prediction_group.add_argument('-rp', '--road_prediction', default=1,choices=range(0,27), type=int, help='road segment to predict')
prediction_group.add_argument('-vpr', '--variable_prediction', default=0, type=int,choices=[0,1,2], help='variable to be predicted')

training_group = parser.add_argument_group('Training parameters')
training_group.add_argument('-ts', '--train_set_size', default=70000, type=int, help='training set size')
training_group.add_argument('-vs', '--valid_set_size', default=30000, type=int, help='validation set size')
training_group.add_argument('-vp', '--valid_partitions', default=100, type=int, help='validation set partitions number')
training_group.add_argument('-tp', '--test_partitions', default=100, type=int, help='test set partitions number')
training_group.add_argument('-b', '--batch_size', default=16, type=int, help='batch size for SGD')
training_group.add_argument('-l', '--learning_rate', default=1e-4, type=float, help='learning rate for SGD')
training_group.add_argument('-dr', '--decay_rate', default=0.1, type=float, help='learning rate decay rate')
training_group.add_argument('-ds', '--decay-steps', default=1000, type=int, help='learning rate decay steps')
training_group.add_argument('-c', '--gradient_clip', default=40.0, type=float, help='clip at this max norm of gradient')
training_group.add_argument('-m', '--max_steps', default=10000, type=int, help='max number of iterations for training')
training_group.add_argument('-s', '--save', action='store_true', help='save the model every epoch')
training_group.add_argument('-ens', '--ensemble', default=1, type=int, help='Number of the model in the ensemble')
training_group.add_argument('-e', '--epochs', default=20, type=int, help='Max epochs')
comparative_group = parser.add_argument_group('Comparative group')
comparative_group.add_argument('-cf', '--comparative_file',type=str, help='file to print final error measurement')
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



var_pred = utils.traff_var(args.variable_prediction)

train_set = (train_set - mean) / stddev
valid_set = (valid_set - mean) / stddev
valid_set2 = (valid_set2 - mean) / stddev
test_set = (test_set - mean) / stddev


print('Training set', train_set.shape, train_labels.shape)
print('Validation set', valid_set.shape, valid_labels.shape)
print('Test set', valid_set2.shape, valid_labels2.shape)

print('Building model...')


train_labels = train_labels[:,args.road_prediction,:,var_pred]
valid_labels = valid_labels[:,args.road_prediction,:,var_pred]
valid_labels2 = valid_labels2[:,args.road_prediction,:,var_pred]
test_labels = test_labels[:,args.road_prediction,:,var_pred]

backend.clear_session()

conv_model = model.ConvTraff(args.forecast_window)


history = utils.compile_and_fit(conv_model,train_set,train_labels, valid_set, valid_labels,
            initial_learning_rate = args.learning_rate,decay_steps = args.decay_steps, 
            decay_rate = args.decay_rate,gradient_clip =args.gradient_clip,max_epochs=args.epochs,
            batch=args.batch_size)

pred   = conv_model.predict(test_set)


conv_model.build_graph().summary()

tf.keras.utils.plot_model(

    conv_model.build_graph(),
    to_file='ConvTraffBase.png', dpi=96,
    show_shapes=True, show_layer_names=True,
    expand_nested=False
)

if args.comparative_file != None:
    with open(args.comparative_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([args.batch_size, args.learning_rate,args.decay_rate,
        history.history['loss'][-1],history.history['mean_absolute_error'][-1],
        history.history['mean_squared_error'][-1],history.history['root_mean_squared_error'][-1]
        ])

print('='*50)
print('loss:' + str(history.history['loss']))
print('mae:' + str(history.history['mean_absolute_error']))
print('mape:' + str(history.history['mean_absolute_percentage_error']))
print('mse:' + str(history.history['mean_squared_error']))
print('rmse:' + str(history.history['root_mean_squared_error']))



#utils.plot_history(history)
utils.plot_prediction(test_labels[150:200], pred[150:200])

print('MAE in test_set:')
print(tf.keras.losses.mean_absolute_error(test_labels.flatten(),pred.flatten()).numpy())

print('MAPE in test_set:')
print(tf.keras.losses.mean_absolute_percentage_error(test_labels.flatten(),pred.flatten()).numpy())