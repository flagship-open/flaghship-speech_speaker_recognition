import numpy as np
import keras
import librosa
import random

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, cropsize=200, flag='fix', batch_size=32, dim=(32,32,32),
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.cropsize = cropsize
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.flag = flag
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
#        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, temp in enumerate(list_IDs_temp):
            ID, start_idx = temp[0], int(temp[1])
            # Store sample
            segment = np.transpose(np.load(ID))
            if self.flag== 'random':
                start_idx = random.randint(0, segment.shape[0]-self.cropsize)
                segment = segment[start_idx:start_idx+self.cropsize]
            else:
                segment = segment[start_idx:start_idx+self.cropsize]
            X[i,] = np.expand_dims(segment, 2)
            # Store class
            y[i] = self.labels[ID]
#        print(X.shape)
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
