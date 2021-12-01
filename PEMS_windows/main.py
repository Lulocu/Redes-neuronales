from matplotlib.pyplot import hist
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import utils
import model


path = '/home/luis/Documentos/pruebaRedes/datasetsMini/'
file_dataset = '2015.csv'
file_test = '2016.csv'
#file_validate = '2016Prueba.csv'
filename_dataset = os.path.join(path,file_dataset)
filename_val = os.path.join(path,file_test)

train_df, val_df, test_df = utils.get_normalised_data(filename_dataset,filename_val)

#Añades al modelo los datasets necesarios
model.WindowGenerator.plot = utils.plot
model.WindowGenerator.make_dataset = utils.make_dataset
model.WindowGenerator.train = utils.train
model.WindowGenerator.val = utils.val
model.WindowGenerator.test = utils.test
model.WindowGenerator.example = utils.example

val_performance = {}
performance = {}


CONV_WIDTH = 36#3 #La ventana de  valores a leer antes de realizar una prediccion, afecta al tamaño de la entrada

conv_window = model.WindowGenerator( #prepara los datos para entrenar el modelo
    input_width=CONV_WIDTH,
    label_width=1,
    shift=1,
    train_df=train_df, val_df=val_df, test_df=test_df,
    label_columns=['flow','density','speed'])


LABEL_WIDTH = 18#24
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = model.WindowGenerator( #Prepara los datos para plotearlos
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1,
    train_df=train_df, val_df=val_df, test_df=test_df,
    label_columns=['flow','density','speed'])



input=keras.Input(shape=(CONV_WIDTH,5))
l1 = layers.Conv1D(filters=32,
                           kernel_size=(CONV_WIDTH,),
                           activation='relu')(input)
l2=layers.Dense(units=32, activation='relu')(l1)
output = layers.Dense(units=3)(l2)
conv_model = keras.Model(inputs=input,outputs = output, name= 'functionalAPI')



history = model.compile_and_fit(conv_model,conv_window,max_epochs=20)

print(history.params)


val_performance['Conv'] = conv_model.evaluate(conv_window.val)
performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)


keras.utils.plot_model(conv_model, "CNNFUN_model.png", show_shapes=True)

#wide_conv_window.plot(plot_col='flow',model=conv_model)
#wide_conv_window.plot(plot_col='density',model=conv_model)
#wide_conv_window.plot(plot_col='speed',model=conv_model)

#utils.plot_mae_validation_loss(history)

utils.plot_mae_mape(history)

print('='*50)
print('VAL_perf:' + str(val_performance['Conv'][1]))
print('TEST_perf:' + str(performance['Conv'][1]))