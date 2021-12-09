import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

#Cargar los datos

'''
La base de datos MNIST está formada por 60 000 imagenes de entrenamiento y 10 000 de test
x_train:array (60 000, 28, 28). Valores de pixeles entre 0 y 255
y_train: array (60 000,) etiquetas del 0-9 para los valores  de x_train
'''
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#Aplano los datos en vectores
'''
x_train:array (60 000, 784). Valores de pixeles entre 0 y 1 porque los hemos normalizado
y_train: array (60 000,) etiquetas del 0-9 para los valores  de x_train
'''
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255



'''
Creación de capas y modelo


'''
inputs = keras.Input(shape=(784,),name = 'Red MNIST')
dense = layers.Dense(64,activation="relu")(inputs)
dense2 = layers.Dense(64,activation="relu")(dense)
outputs = layers.Dense(10)(dense2) #Tiene 10 neuronas de salida porque sale en 10 clases

model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

'''
Comprobaciones del modelo en consola y como png

'''

model.summary()

keras.utils.plot_model(model, "MNIST_model.png", show_shapes=True)

'''
Compilación modelo

'''
print('\nComienza la compilación \n')

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

#Entrenamiento
history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)

#Evaluación
test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])


'''
#Visualizar datos
for i in range(9):
    #clear the image because we didn't close it
    plt.clf()

    #show the image
#    plt.figure(figsize=(5, 5))
    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
    plt.colorbar()
    print("Pausing...")
    plt.pause(5)

'''