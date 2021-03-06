import errno
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.misc
from scipy.ndimage import rotate
from scipy.stats import bernoulli

class Dataset(object):

    DRIVING_LOG_FILE = '../data/data/driving_log.csv'
    IMG_PATH = '../data/data/'
    STEERING_COEFFICIENT = 0.229

    def __init__(self):
        self.X_train = None
        self.X_valid = None
        self.X_test  = None
        self.Y_train = None
        self.Y_valid = None
        self.Y_test  = None

    def next_batch(self, batch_size=64):
        while True:
            X_batch = []
            y_batch = []
            images = self.get_next_image_files(batch_size)

            for img_file, angle in images:
                raw_image = plt.imread(self.IMG_PATH + img_file)
                raw_angle = angle
                new_image, new_angle = self.generate_new_image(raw_image, raw_angle)
                X_batch.append(new_image)
                y_batch.append(new_angle)

            assert len(X_batch) == batch_size, 'len(X_batch) == batch_size should be True'

            yield np.array(X_batch), np.array(y_batch)

    def get_next_image_files(self, batch_size=64):
        data        = pd.read_csv(self.DRIVING_LOG_FILE)
        num_of_img  = len(data)
        rnd_indices = np.random.randint(0, num_of_img, batch_size)

        image_files_and_angles = []

        for index in rnd_indices:

            rnd_image = np.random.randint(0, 3)

            if rnd_image == 0:
                img = data.iloc[index]['left'].strip()
                angle = data.iloc[index]['steering'] + self.STEERING_COEFFICIENT
                image_files_and_angles.append((img, angle))

            elif rnd_image == 1:
                img = data.iloc[index]['center'].strip()
                angle = data.iloc[index]['steering']
                image_files_and_angles.append((img, angle))
            else:
                img = data.iloc[index]['right'].strip()
                angle = data.iloc[index]['steering'] - self.STEERING_COEFFICIENT
                image_files_and_angles.append((img, angle))

        return image_files_and_angles

    def generate_new_image(self, image, steering_angle, top_crop_percent=0.35, bottom_crop_percent=0.1,
                           resize_dim=(64, 64), do_shear_prob=0.9):

        head = bernoulli.rvs(do_shear_prob)

        if head == 1:
            image, steering_angle = self.random_shear(image, steering_angle)

        image = self.crop(image, top_crop_percent, bottom_crop_percent)

        image, steering_angle = self.flip(image, steering_angle)

        image = self.random_gamma(image)

        image = self.resize(image, resize_dim)

        return image, steering_angle

    def random_shear(self, image, steering_angle, shear_range=200):
        rows, cols, ch = image.shape
        dx = np.random.randint(-shear_range, shear_range + 1)
        random_point = [cols / 2 + dx, rows / 2]
        pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
        pts2 = np.float32([[0, rows], [cols, rows], random_point])
        dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
        steering_angle += dsteering

        return image, steering_angle

    def crop(self, image, top_percent, bottom_percent):

        top = int(np.ceil(image.shape[0] * top_percent))
        bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))

        return image[top:bottom, :]

    def flip(self, image, steering_angle, flipping_prob=0.5):
        head = bernoulli.rvs(flipping_prob)
        if head:
            return np.fliplr(image), -1 * steering_angle
        else:
            return image, steering_angle

    def random_gamma(self, image):
        gamma = np.random.uniform(0.4, 1.5)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    def resize(self, image, new_dim):
        return scipy.misc.imresize(image, new_dim)
