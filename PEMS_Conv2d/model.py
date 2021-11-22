from __future__ import print_function
from __future__ import division

import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.layers as layers

class ConvTraff(tf.keras.Model):

    def __init__(self, output_size, training=True):
        super(ConvTraff, self).__init__()
        self.res_1 = Resnet(32, training)
        self.res_2 = Resnet(64, training)
        self.res_3 = Resnet(96, training)
        self.flatten = layers.Flatten()
        self.dense_1 = layers.Dense(2048)
        self.drop = layers.Dropout(.4, training)
        self.dense_2 = layers.Dense(1024)
        self.dense_3 = layers.Dense(output_size, activation=None)


    def call(self, inputs):
        input = keras.Input(inputs)
        x = self.res_1(input)
        x = self.res_1(x)
        x = self.res_1(x)

        x = self.res_2(x)
        x = self.res_2(x)
        x = self.res_2(x)

        x = self.res_3(x)
        x = self.res_3(x)
        x = self.res_3(x)

        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.drop(x)
        x = self.dense_2(x)
        return self.dense_3(x)

class Resnet(keras.layers.Layer):
    def __init__(self,filters, training):
        super(Resnet, self).__init__()
        self.conv = layers.Conv2D(filters,[3,3],strides=[1,1],padding="same")
        self.batch_norm = layers.BatchNormalization(training)
        self.relu = tf.nn.relu(1)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batch_norm(x)
        x = self.conv(x)
        x = self.batch_norm(x)
        return tf.nn.relu(x + inputs)
         