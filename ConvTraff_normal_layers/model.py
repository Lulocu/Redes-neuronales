import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.layers as layers

class ConvTraff(keras.Model):

    def __init__(self, output_size):
        super(ConvTraff, self).__init__()
        self.norm = layers.BatchNormalization()
        self.res32_1 = Resnet(32)
        self.res32_2 = Resnet(32)
        self.res32_3 = Resnet(32)

        self.res64_1 = Resnet(64)
        self.res64_2 = Resnet(64)
        self.res64_3 = Resnet(64)

        self.res96_1 = Resnet(96)
        self.res96_2 = Resnet(96)
        self.res96_3 = Resnet(96)

        self.flatten = layers.Flatten()
        self.dense_1 = layers.Dense(2048)
        self.drop = layers.Dropout(.4)
        self.dense_2 = layers.Dense(1024)
        self.dense_3 = layers.Dense(output_size, activation=None)

        self.padding_1 = layers.ZeroPadding2D([(0,0),(14,15)],data_format= 'channels_first')
        self.padding_2 = layers.ZeroPadding2D([(0,0),(16,16)],data_format= 'channels_first')
        self.padding_3 = layers.ZeroPadding2D([(0,0),(16,16)],data_format= 'channels_first')


    def call(self, inputs):

        x= self.norm(inputs)
        x = self.padding_1(x)
        x = self.res32_1(x)
        x = self.res32_2(x)
        x = self.res32_3(x)


        x = self.padding_2(x)
        x = self.res64_1(x)
        x = self.res64_2(x)
        x = self.res64_3(x)

        x = self.padding_3(x)
        x = self.res96_1(x)
        x = self.res96_2(x)
        x = self.res96_3(x)

        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.drop(x)
        x = self.dense_2(x)
        return self.dense_3(x)
     
    def get_config(self):
        config = super(ConvTraff, self).get_config()
        config.update({
                       "norm": self.norm,
                       "res32_1": self.res32_1,
                       "res32_2": self.res32_2,
                       "res32_3": self.res32_3,

                       "res64_1": self.res64_1,
                       "res64_2": self.res64_2,
                       "res64_3": self.res64_3,

                       "res96_1": self.res96_1,
                       "res96_2": self.res96_2,
                       "res96_3": self.res96_3,

                       "flatten": self.flatten,
                       "dense_1": self.dense_1,
                       "drop": self.drop,
                       "dense_2": self.dense_2,
                       "dense_3": self.dense_3,
                       
                       "padding_1": self.padding_1,
                       "padding_2": self.padding_2,
                       "padding_3": self.padding_3
                       })
        return config


    def build_graph(self,window):
        x = keras.Input(shape=(None, window, 3))
        return keras.Model(inputs=[x], outputs=self.call(x))

    def summary(self):

        return self.build_graph.summary() 


class Resnet(keras.layers.Layer):

    def __init__(self,filters):
        super(Resnet, self).__init__()
        
        self.conv = layers.Conv2D(filters,[3,3],strides=[1,1],padding="same")
        self.batch_norm = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters,[3,3],strides=[1,1],padding="same")
        self.batch_norm2 = layers.BatchNormalization()
        self.sum = layers.Add()
        self.relu1 = layers.ReLU()
        self.relu2 = layers.ReLU()

    def call(self, inputs):
        
        a = self.conv(inputs)
        b = self.batch_norm(a)
        b = self.relu1(b)
        c = self.conv2(b)
        d = self.batch_norm2(c)
        e = self.sum([d,inputs])
        f = self.relu2(e)
        return f

    def get_config(self):
        config = super(Resnet, self).get_config()
        config.update({
                       "conv": self.conv,
                       "batch_norm": self.batch_norm,
                       "conv2": self.conv2,
                       "batch_norm2": self.batch_norm2,
                       "sum": self.sum,
                       "relu1": self.relu1,
                       "relu2": self.relu2,
                       })


        return config