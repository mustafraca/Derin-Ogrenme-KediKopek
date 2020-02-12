# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 19:19:30 2019

@author: ACAR
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import layers


sınıflandırma = Sequential()

# ADIM 1 - Convolution
sınıflandırma.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# ADIM 2- Pooling
sınıflandırma.add(MaxPooling2D(pool_size = (2, 2)))

# İkinci Convolutional Katman
sınıflandırma.add(Conv2D(32, (3, 3), activation = 'relu'))
sınıflandırma.add(MaxPooling2D(pool_size = (2, 2)))

# ADIM 3 - Flattening
sınıflandırma.add(Flatten())

sınıflandırma.add(layers.Dropout(0.5))

# ADIM 4 - Full connection
sınıflandırma.add(Dense(units = 128, activation = 'relu'))

sınıflandırma.add(Dense(units = 1, activation = 'sigmoid'))

sınıflandırma.summary()

# CNN DERLEME
sınıflandırma.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

history = sınıflandırma.fit_generator(training_set,
                         steps_per_epoch = 1000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)

sınıflandırma.save('kedikopek.h5')

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

deneme = 'dataset/deneme'

import numpy as np
from keras.preprocessing import image

def testing_image(image_directory):
    test_image = image.load_img(image_directory, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = sınıflandırma.predict(x = test_image)
    print(result)
    if result[0][0]  == 1:
        prediction = 'Köpek'
    else:
        prediction = 'Kedi'
    return prediction