from __future__ import print_function

from data_loader import load_data
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD
from keras.utils import plot_model
from keras import backend as K
from datetime import datetime
import time

import matplotlib.pyplot as plt

# fix plotting error
import pydot
pydot.find_graphviz = lambda: True

import logging
logging.getLogger().setLevel(logging.DEBUG)

(train_img, train_lbl), (test_img, test_lbl) = load_data()

BATCH_SIZE = 100
NUM_CLASSES = 10
NUM_EPOCHS = 10

input_shape = (train_img.shape[1], train_img.shape[2], train_img.shape[3])

# convert class vectors to binary class matrices
train_lbl = keras.utils.to_categorical(train_lbl, NUM_CLASSES)
test_lbl = keras.utils.to_categorical(test_lbl, NUM_CLASSES)

model = Sequential()
# CONV1->RELU1->POOL1
model.add(Conv2D(filters=20, kernel_size=(5,5), activation='relu', input_shape=input_shape, strides=(1,1), dilation_rate=(1,1)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
# CONV2->RELU2->POOL2
model.add(Conv2D(filters=50, kernel_size=(5,5), activation='relu', strides=(1,1), dilation_rate=(1,1)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
# FLATTEN->FC1->RELU3
model.add(Flatten())
model.add(Dense(units=500, activation='relu'))
# FC2->SOFTMAX
model.add(Dense(units=NUM_CLASSES, activation='softmax'))

# Compile
model.compile(loss=keras.losses.mean_squared_error,
              optimizer=SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False, clipnorm=0.0, clipvalue=0.0),
              metrics=['accuracy'])

# Fit
start_time = time.time()
history = model.fit(train_img, train_lbl,
          batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS,
          verbose=1,
          validation_data=(test_img, test_lbl),
          shuffle=False)

elapsed_time = time.time() - start_time
print('Fit time: ', elapsed_time)

start_time = time.time()
score = model.evaluate(test_img, test_lbl, batch_size=BATCH_SIZE, verbose=1)
elapsed_time = time.time() - start_time
print('Evaluate time: ', elapsed_time)

t = datetime.now()

# summarize history for accuracy
plt.gcf().clear()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
acc_filename = '{}_model_accuracy_{}{}.png'.format(K.backend(), t.hour, t.minute)
plt.savefig(acc_filename)
# summarize history for loss
plt.gcf().clear()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
loss_filename = '{}_model_loss_{}{}.png'.format(K.backend(), t.hour, t.minute)
plt.savefig(loss_filename)

model_filename = '{}_model_{}{}.png'.format(K.backend(), t.hour, t.minute)
plot_model(model, to_file=model_filename, show_shapes=True, show_layer_names=False)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
