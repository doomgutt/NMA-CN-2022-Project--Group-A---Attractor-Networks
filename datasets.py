# Everything about dataset preparation
# Usage:
# > dataset = Dataset_MNIST() # This will construct all data
# > dataset.get_data()        # This will be the numpy arry for further use
# > dataset.show()            # This will visualise an overview of the dataset

import utilities as uti

from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
import imageio as iio
from PIL import Image

class MyBaseDataset(object):
    def __init__(self):
        self.X = np.empty(0, 0)

    def get_shape(self):
        return self.X.shape

    def get_data(self):
        return self.X

    def show(self):
        pass

class Dataset_demoletters(MyBaseDataset):
    def __init__(self):
        img = iio.imread("./images/pixel-arial-11-font-character-map.png")
        self.LETTERS_RESOLUTION = {}
        for i in range(1, 100):
            self.LETTERS_RESOLUTION[i] = Dataset_demoletters._read_parse_image(img, new_size = i)
        self.X = self.LETTERS_RESOLUTION[64]

    @staticmethod
    def _read_parse_image(img, new_size = 64):
        # read an image

        pixel_i = 115
        pixel_j = 110
        new_size = 60
        all_letter_final = np.empty((9 * 9, new_size, new_size), dtype="int8")
        for i in range(9):
            for j in range(9):
                # print(i, j)
                i_offset = 0
                if i >= 3:
                    i_offset = 40
                if i >= 6:
                    i_offset = 80

                subimg = img[i_offset + pixel_i * i : i_offset +pixel_i * (i+1),
                            pixel_j * j : pixel_j * (j+1),
                            :4].astype("uint8")
                new_img = Image.fromarray(subimg)

                new_img = new_img.resize((new_size, new_size))
                new_arr = np.asarray(new_img)
                # print(new_arr[20:30, 20:30, 3])
                new_arr = (new_arr[:, :, 3] > 100) * 2 - 1
                # all_letter_final[i * 9 + j] = new_arr[:, :, 0]
                all_letter_final[i*9+j] = new_arr[:]
        all_letter_final = all_letter_final.reshape(all_letter_final.shape[0], -1)
        return all_letter_final


    def get_data(self, res = 60):
        return self.LETTERS_RESOLUTION[res]


    def show(self, res = 60):
        all_letter_final = self.LETTERS_RESOLUTION[res]
        for i in range(9):
            for j in range(9):
                arr = all_letter_final[9 * i + j]
                ax = plt.subplot(9, 9, i * 9 + j + 1)
                uti.show_letter(arr, ax)

class Dataset_MNIST(MyBaseDataset):
    def __init__(self):
        # GET mnist data
        mnist = fetch_openml(name='mnist_784', as_frame = False)
        self.X = mnist.data

    def show(self, show_first_n = 10):
        for i in range(show_first_n):
            plt.subplot(1, show_first_n, i + 1)
            plt.imshow(self.X[i].reshape(28, 28))


class Dataset_Demyan(MyBaseDataset):
    def __init__(self):
        all_images_dem = np.empty((6, 50 * 50))
        for i in range(6):
            img = iio.imread(f"./images/hand_drawn/dem_{i+1}.jpg")
            print(img.shape)
            new_img = img.copy().astype("int")
            new_img[new_img < 100] = -1
            new_img[new_img >= 100] = 1
            new_img = new_img[::4, ::4]
            all_images_dem[i] = new_img.reshape(-1).copy()
        self.X = all_images_dem

    def show(self):
        fig, axs = plt.subplots(1, 6)
        for i in range(6):
            uti.show_letter(self.X[i], axs[i])
