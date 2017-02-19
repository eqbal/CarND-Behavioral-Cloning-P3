# -*- coding: utf-8 -*-
from __future__ import print_function
import random

import numpy as np
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Lambda
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K

from dataset import *
import errno
import json
import os

class BehavioralCloningNet(object):

    FILE_PATH = 'model.h5'

    def __init__(self):
        self.model = None
        self.number_of_epochs = 8
        self.number_of_samples_per_epoch = 20032
        self.number_of_validation_samples = 6400
        self.learning_rate = 1e-4
        self.activation_relu = 'relu'
        self.dataset = Dataset()


    # Our model is based on NVIDIA's "End to End Learning for Self-Driving Cars" paper
    # Source:  https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    def build_model(self):

        model = Sequential()

        # Normalize the data
        model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3)))

        # starts with five convolutional and maxpooling layers
        model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
        model.add(Activation(self.activation_relu))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

        model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
        model.add(Activation(self.activation_relu))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

        model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
        model.add(Activation(self.activation_relu))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

        model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
        model.add(Activation(self.activation_relu))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

        model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
        model.add(Activation(self.activation_relu))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

        model.add(Flatten())

        model.add(Dense(1164))
        model.add(Activation(self.activation_relu))

        model.add(Dense(100))
        model.add(Activation(self.activation_relu))

        model.add(Dense(50))
        model.add(Activation(self.activation_relu))

        model.add(Dense(10))
        model.add(Activation(self.activation_relu))

        model.add(Dense(1))

        model.summary()

        self.model = model

    def compile(self):

        self.model.compile(optimizer=Adam(self.learning_rate), loss="mse")

        # let's train the model using SGD + momentum (how original :P ). 
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # self.model.compile(loss='categorical_crossentropy',
                           # optimizer=sgd,
                           # metrics=['accuracy'])

    def augmentation(self, dataset):

        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(
                featurewise_center=False,             # set input mean to 0 over the dataset
                samplewise_center=False,              # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,   # divide each input by its std
                zca_whitening=False,                  # apply ZCA whitening
                horizontal_flip=True,                 # randomly flip images
                vertical_flip=False)                  # randomly flip images

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(dataset.X_train)
        self.datagen = datagen


    def train(self):
        print('Start training.')

        train_batch      = self.dataset.next_batch()
        validation_batch = self.dataset.next_batch()

        history = self.model.fit_generator(
                train_batch,
                samples_per_epoch=self.number_of_samples_per_epoch,
                nb_epoch=self.number_of_epochs,
                validation_data=validation_batch,
                nb_val_samples=self.number_of_validation_samples,
                verbose=1
            )

        self.history = history

    def save(self, model_name='model.json', weights_name='model.h5'):
        print('Model Saved.')
        self.delete_file(model_name)
        self.delete_file(weights_name)

        json_string = self.model.to_json()

        with open(model_name, 'w') as outfile:
            json.dump(json_string, outfile)

        self.model.save_weights(weights_name)

    def load(self, file_path=FILE_PATH):
        print('Model Loaded.')
        self.model = load_model(file_path)

    def predict(self, image):
        image = image.astype('float32')
        result = self.model.predict_proba(image)
        print(result)
        result = self.model.predict_classes(image)

        return result[0]

    def delete_file(self, file):
        try:
            os.remove(file)

        except OSError as error:
            if error.errno != errno.ENOENT:
                raise

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.X_test, dataset.Y_test, verbose=0)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

