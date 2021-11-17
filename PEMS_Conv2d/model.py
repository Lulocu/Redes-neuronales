from __future__ import print_function
from __future__ import division

import tensorflow.keras as keras
import tensorflow.keras.layers as layers


def model(x, mean, stddev, output_size, is_training=True, reuse=None):
    net = (x - mean) / stddev  # Normalization of input


    input_res1=layers.Input(shape = keras.shape(net))
    x = layers.Conv2D(32,[3,3],[1,1])(input_res1) #Tal vez falta el scope,scope=scope + '_1'
    x = layers.BatchNormalization()(x)#No parece muy igual
    x = layers.Conv2D(32,[3,3],[1,1])(x)
    x = layers.BatchNormalization()(x)
    output_res1 = layers.ReLU(x)


    x = layers.Conv2D(32,[3,3],[1,1])(output_res1) #Tal vez falta el scope,scope=scope + '_1'
    x = layers.BatchNormalization()(x)#No parece muy igual
    x = layers.Conv2D(32,[3,3],[1,1])(x)
    x = layers.BatchNormalization()(x)
    output_res2 = layers.ReLU(x)


    x = layers.Conv2D(32,[3,3],[1,1])(output_res2) #Tal vez falta el scope,scope=scope + '_1'
    x = layers.BatchNormalization()(x)#No parece muy igual
    x = layers.Conv2D(32,[3,3],[1,1])(x)
    x = layers.BatchNormalization()(x)
    output_res3 = layers.ReLU(x)


    x = layers.Conv2D(64,[3,3],[1,1])(output_res3) #Tal vez falta el scope,scope=scope + '_1'
    x = layers.BatchNormalization()(x)#No parece muy igual
    x = layers.Conv2D(64,[3,3],[1,1])(x)
    x = layers.BatchNormalization()(x)
    output_res4 = layers.ReLU(x)


    x = layers.Conv2D(64,[3,3],[1,1])(output_res4) #Tal vez falta el scope,scope=scope + '_1'
    x = layers.BatchNormalization()(x)#No parece muy igual
    x = layers.Conv2D(64,[3,3],[1,1])(x)
    x = layers.BatchNormalization()(x)
    output_res5 = layers.ReLU(x)


    x = layers.Conv2D(64,[3,3],[1,1])(output_res5) #Tal vez falta el scope,scope=scope + '_1'
    x = layers.BatchNormalization()(x)#No parece muy igual
    x = layers.Conv2D(64,[3,3],[1,1])(x)
    x = layers.BatchNormalization()(x)
    output_res6 = layers.ReLU(x)


    x = layers.Conv2D(96,[3,3],[1,1])(output_res6) #Tal vez falta el scope,scope=scope + '_1'
    x = layers.BatchNormalization()(x)#No parece muy igual
    x = layers.Conv2D(96,[3,3],[1,1])(x)
    x = layers.BatchNormalization()(x)
    output_res7 = layers.ReLU(x)


    x = layers.Conv2D(96,[3,3],[1,1])(output_res7) #Tal vez falta el scope,scope=scope + '_1'
    x = layers.BatchNormalization()(x)#No parece muy igual
    x = layers.Conv2D(96,[3,3],[1,1])(x)
    x = layers.BatchNormalization()(x)
    output_res8 = layers.ReLU(x)


    x = layers.Conv2D(96,[3,3],[1,1])(output_res8) #Tal vez falta el scope,scope=scope + '_1'
    x = layers.BatchNormalization()(x)#No parece muy igual
    x = layers.Conv2D(96,[3,3],[1,1])(x)
    x = layers.BatchNormalization()(x)
    output_res9 = layers.ReLU(x)


    net = layers.flatten(output_res9)

    net = layers.fully_connected(net, 2048)
    net = layers.dropout(net, keep_prob=0.6, )
    net = layers.fully_connected(net, 1024)
    out = layers.fully_connected(net, output_size, activation_fn=None)

    model = keras.model(input,out,name='PruebaConvTraff')

