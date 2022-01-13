import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import pickle
from enum import IntEnum
from tensorflow import keras
import datetime
from memory_profiler import profile



def get_dataset_name(time_window, time_aggregation, forecast_window, forecast_aggregation, train_set_size,
                     valid_set_size):
    pickle_filename = 'dataset_'
    pickle_filename += str(time_window) + '_'
    pickle_filename += str(time_aggregation) + '_'
    pickle_filename += str(forecast_window) + '_'
    pickle_filename += str(forecast_aggregation) + '_'
    pickle_filename += str(train_set_size) + '_'
    pickle_filename += 'norm' + '_'
    pickle_filename += str(valid_set_size) + '.pickle'

    return pickle_filename


def parse_csv_file(filename, time_window, time_aggregation, forecast_window, forecast_aggregation):
    print('\tParsing', filename)

    timesteps = set()
    sections = set()
    data = []

    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='\"')
        next(reader)
        for row in reader:
            timesteps.add(int(row[0]))
            sections.add(int(row[1]))
            data.append(row[2:])

    data = np.asarray(data, dtype=np.float32)
    num_sections = max(sections) + 1
    num_timesteps = max(timesteps) + 1

    sequence = []

    for i in range(num_timesteps):
        stack = None
        for j in range(num_sections):
            stack = np.vstack([stack, data[i * num_sections + j]]) if stack is not None else data[i * num_sections + j]

        sequence.append(stack)

    d = []
    l = []

    max_timestep = num_timesteps - time_window * time_aggregation - forecast_window * forecast_aggregation + 1
    for i in range(0, max_timestep, time_aggregation):
        time_steps = []
        for j in range(time_window):
            initial_index = i + j * time_aggregation
            final_index = i + (j + 1) * time_aggregation
            time_steps.append(np.mean(np.stack(sequence[initial_index:final_index], axis=1), axis=1))
        d.append(np.stack(time_steps, axis=1))
        forecast_steps = []
        for j in range(forecast_window):
            initial_index = i + time_window + j * forecast_aggregation
            final_index = i + time_window + (j + 1) * forecast_aggregation
            forecast_steps.append(np.mean(np.stack(sequence[initial_index:final_index], axis=1), axis=1))
        l.append(np.stack(forecast_steps, axis=1))

    return d, l


def get_dataset(pickle_filename, args, parser):
    if os.path.exists(pickle_filename):
        print('Loading dataset from ' + pickle_filename + '...')

        with open(pickle_filename, 'rb') as f:
            save = pickle.load(f)
            valid_set = save['valid_set']
            valid_labels = save['valid_labels']
            valid_set2 = save['valid_set2']
            valid_labels2 = save['valid_labels2']
            test_set = save['test_set']
            test_labels = save['test_labels']
            #mean = save['mean']
            #stddev = save['stddev']
            f.close()

        train_set = np.load('train_set_norm.npy')
        train_labels = np.load('train_labels_norm.npy')
    else:
        if args.datasets is None or args.test_set is None:
            print('Dataset not found. You must give dataset and test set arguments from command line.')
            parser.print_help()
            exit()

        print('Generating training, validation and test sets...')

        dataset = []
        labels = []
        valid_set = []
        valid_labels = []
        test_set = []
        test_labels = []

        for dataset_file in args.datasets:
            ds, lb = parse_csv_file(dataset_file, args.time_window, args.time_aggregation, args.forecast_window,
                                    args.forecast_aggregation)
            dataset += ds
            labels += lb

        for valid_set_file in args.valid_set:
            ds, lb = parse_csv_file(valid_set_file, args.time_window, args.time_aggregation, args.forecast_window,
                                    args.forecast_aggregation)
            valid_set += ds
            valid_labels += lb

        for test_set_file in args.test_set:
            ds, lb = parse_csv_file(test_set_file, args.time_window, args.time_aggregation, args.forecast_window,
                                    args.forecast_aggregation)
            test_set += ds
            test_labels += lb

        del ds, lb

        permutation = np.random.permutation(len(dataset))
        dataset = np.asarray(dataset)[permutation]
        labels = np.asarray(labels)[permutation]
        permutation = np.random.permutation(len(valid_set) + len(test_set))
        valid_set = np.asarray(valid_set + test_set)[permutation]
        valid_labels = np.asarray(valid_labels + test_labels)[permutation]
        test_set = np.asarray(test_set)
        test_labels = np.asarray(test_labels)

        #mean = np.mean(dataset, axis=(0, 1, 2))
        #stddev = np.std(dataset, axis=(0, 1, 2))

        train_set = dataset[:args.train_set_size]
        train_labels = labels[:args.train_set_size]
        valid_set2 = valid_set[args.valid_set_size:2 * args.valid_set_size]
        valid_labels2 = valid_labels[args.valid_set_size:2 * args.valid_set_size]
        valid_set = valid_set[:args.valid_set_size]
        valid_labels = valid_labels[:args.valid_set_size]
        test_set = test_set[:args.valid_set_size]
        test_labels = test_labels[:args.valid_set_size]

        print('Saving dataset into ' + pickle_filename + '...')

        save = {
            'valid_set': valid_set,
            'valid_labels': valid_labels,
            'valid_set2': valid_set2,
            'valid_labels2': valid_labels2,
            'test_set': test_set,
            'test_labels': test_labels,
            #'mean': mean,
            #'stddev': stddev
        }

        f = open(pickle_filename, 'wb')
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()

        np.save('train_set_norm.npy', train_set)
        np.save('train_labels_norm.npy', train_labels)

    del save
    return (train_set, train_labels, valid_set, valid_labels, valid_set2, valid_labels2, test_set, test_labels)#, mean,
            #stddev)
            
class traff_var(IntEnum):
    FLOW = 0
    OCCUPANCY = 1
    SPEED = 2

def l2loss(y_true, y_pred):
    return tf.nn.l2_loss(y_pred - y_true)


def compile_and_fit(model, train_set,train_labels,valid_set, valid_labels, initial_learning_rate, decay_steps, 
            decay_rate,gradient_clip,batch, max_epochs = 20):

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,
        write_images=True, write_steps_per_second=True,embeddings_freq=1)

    csv_logger = keras.callbacks.CSVLogger('logs/ConvTraff_normal_layers.csv',append =True)

    checkpoint_filepath = 'savedModel/'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='l2_loss',
        mode='auto',
        save_freq='epoch',
        save_best_only=False)

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps, decay_rate, staircase=True)

    model.compile(loss=l2loss,#tf.losses.MeanAbsoluteError(),
                    optimizer= keras.optimizers.SGD(learning_rate,clipnorm = gradient_clip,momentum = 0.9),
                    metrics=[tf.keras.metrics.MeanAbsoluteError(), 
                    tf.keras.metrics.MeanAbsolutePercentageError(),
                    tf.keras.metrics.MeanSquaredError(),
                    tf.keras.metrics.RootMeanSquaredError()])

    history = model.fit(train_set,train_labels,validation_data = (valid_set, valid_labels), 
        batch_size = batch, epochs= max_epochs,shuffle=True,verbose =2,
        callbacks=[csv_logger])#,callbacks=[tensorboard_callback]) #, callbacks=[model_checkpoint_callback]
    return history

def plot_history(history):

    hist = history.history
    epochs = history.epoch
    for metric in hist.keys():
        plt.plot(epochs,hist[metric],marker='o', linestyle='--', color='r', label=metric)
        
        plt.title('Training ' + metric)
        plt.legend()
        plt.xticks(epochs,epochs)
        plt.show()

        
def plot_prediction(real_data, prediction,dataset_len,test_len):

    for i in range(real_data.shape[-1]):
        plt.plot(range(len(real_data[:,i])),real_data[:,i].flatten(),marker='o', linestyle='--', color='r', label="real data")
        plt.plot(range(len(prediction[:,i])),prediction[:,i].flatten(),marker='o', linestyle='-.', color='b', label="prediction")
        plt.title('Compare prediction and real ground on instant' + str(i))
        plt.legend()
        plt.xticks(range(len(real_data)))
        plt.savefig('Images/ConvTraff_normal_layers/Grafica'+str(dataset_len)+'_'+str(test_len)+'_' + str(i) + '.png')