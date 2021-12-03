from matplotlib.pyplot import hist
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

from tensorflow.python.keras.engine import training
import utils
import model
import argparse

#path = '/home/luis/Documentos/pruebaRedes/datasetsMini/'
#file_dataset = '2015.csv'
#file_test = '2016.csv'
##file_validate = '2016Prueba.csv'
#filename_dataset = os.path.join(path,file_dataset)
#filename_val = os.path.join(path,file_test)
#
#train_df, val_df, test_df = utils.get_normalised_data(filename_dataset,filename_val)

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

#train_set,valid_set,valid_set2 = utils.get_normalised_data(args.datasets[0],args.valid_set[0])

var_pred = 0

train_labels = train_labels[:,:,:,var_pred]
valid_labels = valid_labels[:,:,:,var_pred]
valid_labels2 = valid_labels2[:,:,:,var_pred]
test_labels = test_labels[:,:,:,var_pred]

val_performance = {}
performance = {}



training = True
input=keras.Input(shape=(27,12,3))
inputs = tf.pad(input, [[0, 0], [0, 0], [0, 0], [14, 15]])
l1 = model.Resnet(32,training=True)(inputs)
l1 = tf.pad(l1, [[0, 0], [0, 0], [0, 0], [16, 16]])
l1=model.Resnet(64,training=True)(l1)
l2=layers.Dense(units=1)(l1)
conv_model = keras.Model(inputs=input,outputs = l2, name= 'functionalAPI')

#input=keras.Input(shape=(27,12,3))
conv_model.run_eagerly = True
#a = layers.Conv2D(32,[3,3],strides=[1,1],padding="same")(input)
#b = layers.BatchNormalization(training)(a)
#c = layers.Conv2D(32,[3,3],strides=[1,1],padding="same")(b)
#x = layers.BatchNormalization(training)(c)
#outputs = layers.Dense(1, activation=None)(x)

#conv_model = keras.Model(inputs=input,outputs = outputs, name= 'functionalAPI')

conv_model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.SGD(),
                metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsolutePercentageError(),
                tf.keras.metrics.MeanSquaredError(),tf.keras.metrics.RootMeanSquaredError()])

history = conv_model.fit(train_set,train_labels)


keras.utils.plot_model(conv_model, "CNNFUN_model.png", show_shapes=True)

print(history.params)
#wide_conv_window.plot(plot_col='flow',model=conv_model)
#wide_conv_window.plot(plot_col='density',model=conv_model)
#wide_conv_window.plot(plot_col='speed',model=conv_model)

#utils.plot_mae_validation_loss(history)

#utils.plot_mae_mape(history)

#print('='*50)
#print('VAL_perf:' + str(val_performance['Conv'][1]))
#print('TEST_perf:' + str(performance['Conv'][1]))
print('OK')