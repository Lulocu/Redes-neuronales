import numpy as np
import csv
import os
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf


def parse_csv_file(filename, time_window=12, time_aggregation=1, forecast_window=1, forecast_aggregation=1):
    """
    Crea un objeto de tipo Electrodomestico.
    Args:
        filename (string): nombre del fichero que almacena los datos

    Returns:
        train_set
        train_labels
        valid_set
        valid_labels
        valid_set2
        valid_labels2
        test_set
        test_labels
        mean
        stddev
    """
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


def prepare_dataset(filename_dataset, filename_test,filename_validate,train_set_size=70000,valid_set_size=30000):
    """
    Prepares dataset
    Args:
        filename_dataset (string): name of the train dataset file
        filename_test (string): name of the validate dataset file
    Returns:
        train_set
        train_labels
        valid_set
        valid_labels
        valid_set2
        valid_labels2
        test_set
        test_labels
        mean
        stddev
    """
    dataset_file = os.path.splitext(filename_dataset)[0]
    print('\n\n\nData set'+dataset_file)
    pickle_filename = dataset_file+'.pickle'
    print(pickle_filename)

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

        train_set = np.load('train_set.npy')[:2]
        train_labels = np.load('train_labels.npy')[:2]
    else:
        print('Generating training, validation and test sets...')

        dataset = []
        labels = []
        valid_set = []
        valid_labels = []
        test_set = []
        test_labels = []

        
        ds, lb = parse_csv_file(filename_dataset)
        dataset += ds
        labels += lb

        
        ds, lb = parse_csv_file(filename_validate)
        valid_set += ds
        valid_labels += lb

        
        ds, lb = parse_csv_file(filename_test)
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

        train_set = dataset[:train_set_size]
        train_labels = labels[:train_set_size]
        valid_set2 = valid_set[valid_set_size:2 * valid_set_size]
        valid_labels2 = valid_labels[valid_set_size:2 * valid_set_size]
        valid_set = valid_set[:valid_set_size]
        valid_labels = valid_labels[:valid_set_size]
        test_set = test_set[:valid_set_size]
        test_labels = test_labels[:valid_set_size]

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
      shuffle=True,
      batch_size=32,)

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