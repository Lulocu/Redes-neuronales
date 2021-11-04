import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import csv
import os

from tensorflow.python.keras.layers.core import Flatten
import utils

path = '/home/luis/Documentos/pruebaRedes/datasetsMini/'
file_dataset = '2015.csv'
file_test = '2016Prueba.csv'
file_validate = '2016Prueba.csv'
filename_dataset = os.path.join(path,file_dataset)
filename_test = os.path.join(path,file_test)
filename_validate = os.path.join(path,file_validate)
train_set, train_labels, valid_set, valid_labels, valid_set2, \
    valid_labels2, test_set, test_labels, mean,stddev =utils.prepare_dataset(filename_dataset, filename_test, filename_validate)

train_set = train_set.reshape(70000, 972).astype("float32")
train_set = train_set.reshape(30000, 972).astype("float32")

inputs = keras.Input(shape=(972,),name = 'Red MLP PEMS')
dense = layers.Dense(3,activation="relu")(inputs)
dense2 = layers.Dense(1,activation="relu")(dense)
outputs = layers.Dense(units=3)(dense2) #Tiene 10 neuronas de salida porque sale en 10 clases

model = keras.Model(inputs=inputs, outputs=outputs, name="PEMS_model")

'''
Comprobaciones del modelo en consola y como png

'''

model.summary()

keras.utils.plot_model(model, "PEMS_model.png", show_shapes=True)

'''
Compilación modelo

'''
print('\nComienza la compilación \n')

model.compile(
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

#Entrenamiento
history = model.fit(train_set, train_labels, batch_size=1024, epochs=1, validation_split=0.2)

#Evaluación
test_scores = model.evaluate(valid_set, valid_labels, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

aux = model.predict(train_set)

print('\n\n\n')
print(aux)







    



'''
inputs = keras.Input(shape=(27,),name = 'Red PEMS, Perceptrón')
dense = layers.Dense(64,activation="relu")(inputs)
dense2 = layers.Dense(64,activation="relu")(dense)
outputs = layers.Dense(1)(dense2) #Tiene 10 neuronas de salida porque sale en 10 clases

model = keras.Model(inputs=inputs, outputs=outputs, name="pems_model")


model.summary()

keras.utils.plot_model(model, "PEMS_model.png", show_shapes=True)


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