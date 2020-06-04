import os
import numpy as np
import random
from imageio import imread
import tensorflow as tf




class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, file_names, batch_size=32, dim=(480, 640), n_channels=3,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.file_names = file_names
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.file_names) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_file_names = self.file_names[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, Y = self.__data_generation(batch_file_names)

        return X, Y
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            random.shuffle(self.file_names)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, *self.dim, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = imread(ID[0])
            Y[i,] = imread(ID[1])


        return X, Y


# Get the list of all files in directory tree at given path






dirName = './data/nyu2_train'

trainFileList = list()
for (dirpath, dirnames, filenames) in os.walk(dirName):
    for file in filenames:
        if file.endswith('.jpg'):
            trainFileList.append((os.path.join(dirpath, file), os.path.join(dirpath, file)[:-3] + 'png'))
print(len(trainFileList))

random.shuffle(trainFileList)

train_generator = DataGenerator(trainFileList)


dirName = './data/nyu2_test'

testFileList = list()
for (dirpath, dirnames, filenames) in os.walk(dirName):
    for file in filenames:
        if file.endswith('colors.png'):
            testFileList.append((os.path.join(dirpath, file), os.path.join(dirpath, file)[:-10] + 'depth.png'))
print(len(testFileList))

random.shuffle(testFileList)


test_generator = DataGenerator(testFileList)


