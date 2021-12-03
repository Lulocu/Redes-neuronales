import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.layers as layers

class ConvTraff(keras.Model):

    def __init__(self, output_size, training=True):
        super(ConvTraff, self).__init__()
        self.res32_1 = Resnet(32, training)
        self.res32_2 = Resnet(32, training)
        self.res32_3 = Resnet(32, training)

        self.res64_1 = Resnet(64, training)
        self.res64_2 = Resnet(64, training)
        self.res64_3 = Resnet(64, training)

        self.res96_1 = Resnet(96, training)
        self.res96_2 = Resnet(96, training)
        self.res96_3 = Resnet(96, training)

        self.flatten = layers.Flatten()
        self.dense_1 = layers.Dense(2048)
        self.drop = layers.Dropout(.4)
        self.dense_2 = layers.Dense(1024)
        self.dense_3 = layers.Dense(output_size, activation=None)


    def call(self, inputs):

        input= layers.ZeroPadding2D([(0,0),(14,15)],data_format= 'channels_first')(inputs)
        x = self.res32_1(input)
        x = self.res32_2(x)
        x = self.res32_3(x)


        x= layers.ZeroPadding2D([(0,0),(16,16)],data_format= 'channels_first')(x)
        x = self.res64_1(x)
        x = self.res64_2(x)
        x = self.res64_3(x)

        x= layers.ZeroPadding2D([(0,0),(16,16)],data_format= 'channels_first')(x)
        x = self.res96_1(x)
        x = self.res96_2(x)
        x = self.res96_3(x)

        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.drop(x)
        x = self.dense_2(x)
        return self.dense_3(x)
    
    def summary(self):
        x = keras.Input(shape=(27, 12, 3))
        model = keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary() 
         
    def build_graph(self):
        x = keras.Input(shape=(27, 12, 3))
        return keras.Model(inputs=[x], outputs=self.call(x))


class Resnet(keras.layers.Layer):

    def __init__(self,filters, training=True):
        super(Resnet, self).__init__()
        
        self.conv = layers.Conv2D(filters,[3,3],strides=[1,1],padding="same")
        self.batch_norm = layers.BatchNormalization(training)
        self.conv2 = layers.Conv2D(filters,[3,3],strides=[1,1],padding="same")
        self.batch_norm2 = layers.BatchNormalization(training)

    def call(self, inputs):
        
        a = self.conv(inputs)
        b = self.batch_norm(a)
        b = tf.nn.relu(b)   #NuevaAdicion       
        c = self.conv2(b)
        x = self.batch_norm2(c)
        return tf.nn.relu(x + inputs)

