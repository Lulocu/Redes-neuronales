import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os
import pickle

def get_normalised_data(train_df,valid_df,test_df = None):
    """Gets the route to data's files andf returns train set, validate set and test set"""
    train_df_npy = os.path.splitext(train_df)[0] + '.npy'

    if os.path.exists(train_df_npy):
        valid_df_npy = os.path.splitext(valid_df)[0]+ '.npy'
        test_df_npy = os.path.splitext(test_df)[0]+ '.npy'

        train_df = np.load(train_df_npy)
        val_df = np.load(valid_df_npy)
        test_df =np.load(test_df_npy)
    
    else:
        train_df = pd.read_csv(train_df)
        valid_file = pd.read_csv(valid_df)

        if test_df != None:
            test_file = pd.read_csv(test_df)
            val_df = valid_file[:]
            test_df = test_file[:]

        else:
            nt = len(valid_file)
            valid_file = pd.read_csv(valid_df)
            val_df = valid_file[int(nt*0.7):]
            test_df = valid_file[:int(nt*0.7)]


        train_mean = train_df.mean()
        train_std = train_df.std()
        #train_df = (train_df - train_mean) / train_std
        #val_df = (val_df - train_mean) / train_std
        #test_df = (test_df - train_mean) / train_std

        np.save('train_set.npy',train_df)
        np.save('valid_set.npy',val_df)
        np.save('test_set.npy',test_df)

    return train_df, val_df, test_df

def plot(self, plot_col, model=None, max_subplots=3):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
      plt.subplot(max_n, 1, n+1)
      plt.ylabel(f'{plot_col} [normed]')
      plt.plot(self.input_indices, inputs[n, :, plot_col_index],
               label='Inputs', marker='.', zorder=-10)
  
      if self.label_columns:
        label_col_index = self.label_columns_indices.get(plot_col, None)
      else:
        label_col_index = plot_col_index
  
      if label_col_index is None:
        continue
  
      plt.scatter(self.label_indices, labels[n, :, label_col_index],
                  edgecolors='k', label='Labels', c='#2ca02c', s=64)
      if model is not None:
        predictions = model(inputs)
        plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                    marker='X', edgecolors='k', label='Predictions',
                    c='#ff7f0e', s=64)
  
      if n == 0:
        plt.legend()
  
    plt.xlabel('Time [h]')      
    plt.show()      

def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
          data=data,
          targets=None,
          sequence_length=self.total_window_size,
          sequence_stride=1,
          shuffle=False,
          batch_size=8,)#32,)
    ds = ds.map(self.split_window)

    return ds

@property
def train(self):
    return self.make_dataset(self.train_df)

@property
def val(self):
    return self.make_dataset(self.val_df)

@property
def test(self):
    return self.make_dataset(self.test_df)

@property
def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
        # No example batch was found, so get one from the `.train` dataset
        result = next(iter(self.train))
        # And cache it for next time
        self._example = result
    return result


def plot_mae_validation_loss(history):
    mae = history.history['mean_absolute_error']
    val_mae = history.history['val_mean_absolute_error']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(mae) + 1)

    plt.plot(epochs, mae, 'bo', label='Training mae')
    plt.plot(epochs, val_mae, 'b', label='Validation mae')
    plt.title('Training and validation mae')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

def plot_mae_mape(history):
    mae = history.history['mean_absolute_error']
    val_mae = history.history['val_mean_absolute_error']
    mape = history.history['mean_absolute_error']
    val_mape = history.history['val_mean_absolute_percentage_error']

    epochs = range(1, len(mae) + 1)

    plt.plot(epochs, mae, 'bo', label='Training mae')
    plt.plot(epochs, val_mae, 'b', label='Validation mae')
    plt.title('Training and validation mae')
    plt.legend()

    plt.figure()

    plt.plot(epochs, mape, 'bo', label='Training mape')
    plt.plot(epochs, val_mape, 'b', label='Validation mape')
    plt.title('Training and validation mape')
    plt.legend()

    plt.show()
def get_dataset_name(time_window, time_aggregation, forecast_window, forecast_aggregation, train_set_size,
                     valid_set_size):
    pickle_filename = 'dataset_'
    pickle_filename += str(time_window) + '_'
    pickle_filename += str(time_aggregation) + '_'
    pickle_filename += str(forecast_window) + '_'
    pickle_filename += str(forecast_aggregation) + '_'
    pickle_filename += str(train_set_size) + '_'
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
            mean = save['mean']
            stddev = save['stddev']
            f.close()

        train_set = np.load('train_set.npy')
        train_labels = np.load('train_labels.npy')
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

        mean = np.mean(dataset, axis=(0, 1, 2))
        stddev = np.std(dataset, axis=(0, 1, 2))

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
            'mean': mean,
            'stddev': stddev
        }

        f = open(pickle_filename, 'wb')
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()

        np.save('train_set.npy', train_set)
        np.save('train_labels.npy', train_labels)

    del save
    return (train_set, train_labels, valid_set, valid_labels, valid_set2, valid_labels2, test_set, test_labels, mean,
            stddev)

