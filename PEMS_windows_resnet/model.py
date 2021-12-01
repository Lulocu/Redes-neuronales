import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
               train_df, val_df, test_df,
               label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
          self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
            
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])


        return inputs, labels



def compile_and_fit(model, window, patience=2,max_epochs = 20):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.keras.losses.MeanAbsolutePercentageError(),
                tf.keras.losses.MeanAbsoluteError()])#tf.metrics.MeanAbsoluteError(),

    history = model.fit(window.train, epochs=max_epochs,
                      validation_data=window.val,
                      callbacks=[early_stopping])
    return history

class Resnet(keras.layers.Layer):
    def __init__(self,filters, training):
        super(Resnet, self).__init__()
        self.conv = layers.Conv2D(filters,[3,3],strides=[1,1],padding="same")
        self.batch_norm = layers.BatchNormalization(training)
        self.conv_2 = layers.Conv2D(filters,[3,3],strides=[1,1],padding="same")
        self.batch_norm2 = layers.BatchNormalization(training)

    def call(self, inputs):
        
        print('1'*50)
        x = self.conv(inputs)
        print('2'*50)
        x = self.batch_norm(x)
        print('3'*50)
        x = self.conv_2(x)
        print('4'*50)
        x = self.batch_norm2(x)
        print('5'*50)
        return tf.nn.relu(x + inputs)