from keras import backend as K
from keras.layers import Dense, Activation, Reshape, Lambda, BatchNormalization, Input, Conv2D, MaxPooling2D
from keras.layers import Permute, Lambda, Multiply, Dropout, Flatten, TimeDistributed, Embedding
from keras.layers.recurrent import LSTM
from keras.layers.merge import add, concatenate
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras_layer_normalization import LayerNormalization
from sklearn.model_selection import train_test_split
from keras.backend.tensorflow_backend import set_session, clear_session
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras.utils.io_utils import HDF5Matrix
import keras
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import datetime
import os
import h5py
from best_model.amsoftmax import *
from dataloader import DataGenerator


def build_CRNN_200(input_shape):

    # Input layer
    inputs = Input(name='input_1', shape=input_shape, dtype='float32')  # (None, 200, 40, 1)

    # Convolution layer
    inner = Conv2D(8, (5, 5), padding='same', name='conv2d_1', strides = (1,1), activation = 'relu', kernel_initializer='he_normal')(inputs)  # (None, 200, 40, 8)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_1')(inner)  # (None, 100, 20, 8)
    inner = Dropout(0.5, name='dropout_1')(inner)

    inner = Conv2D(16, (5, 5), padding='same', name='conv2d_2', strides = (1,1), activation = 'relu', kernel_initializer='he_normal')(inner)  # (None, 100, 20, 16)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_2')(inner)  # (None, 50, 10, 16)
    inner = Dropout(0.5, name='dropout_2')(inner)

    inner = Conv2D(32, (5, 5), padding='same', name='conv2d_3', strides = (1,1), activation = 'relu', kernel_initializer='he_normal')(inner)  # (None, 50, 10, 32)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_3')(inner)  # (None, 25, 5, 32)
    inner = Dropout(0.5, name='dropout_3')(inner)

    # Flatten
    # inner = TimeDistributed(Dense(_f))(inner)
    inner = TimeDistributed(Flatten(),name='time_distributed_1')(inner)  # (None, 25, 160)

    # LSTM layer
    lstm_1 = LSTM(64, input_shape = (25, 160),return_sequences=True, kernel_initializer='he_normal', name='lstm_1')(inner)  # (None, 62, 64)
    lstm_1 = LayerNormalization()(lstm_1)

    # Attention Layer
    inner = Permute((2, 1),input_shape = (25, 64), name='permute_1')(lstm_1)  # (None, 64, 25)
    inner = Reshape((64, 25), name='reshape_1')(inner)  # (None, 64, 25)
    inner = Dense(25, kernel_initializer='he_normal',name='dense_1')(inner) # (None, 64, 25)

    inner = Permute((2, 1), input_shape = (64, 25), name='permute_2')(inner)  # (None, 25, 64)
    inner = Multiply(name='multiply_1')([lstm_1,inner]) # (None, 25, 64)
    inner = Lambda(lambda xin: K.sum(xin, axis=1))(inner)

    # Output
    predictions = Dense(units=921, name='output_layer')(inner) # (None, 921)

    model = Model(inputs=inputs, outputs=predictions)
    return model


def train_model():
    ## GPU Setting  : 30%
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    config = tf.ConfigProto(device_count = {'GPU': 1})
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    session = tf.Session(config=config)
    set_session(session)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    ## Parameter
    epochs = 150
    batch_size = 64
    cropsize = 200
    input_shape = (cropsize, 40, 1)

    print('input_shape = ', input_shape)


    # Parameters
    params = {'dim': input_shape,
              'batch_size': 64,
              'n_classes': 921,
              'shuffle': True}
    
    # Datasets
    with open('./data/train.txt', 'r') as fp:
        train_list = fp.readlines()
    
    with open('./data/val.txt', 'r') as fp:
        val_list = fp.readlines()

    with open('./data/test.txt', 'r') as fp:
        test_list = fp.readlines()
    
    file_list = train_list + val_list + test_list
    labels = dict()
    for i in range(len(file_list)):
        label, filename, _ = file_list[i].replace('\n', '').split('\t')
        labels[filename] = int(label)
    
    partition = dict()# IDs
    partition['train'] = []
    for i in range(len(train_list)):
        label, filename, start_idx = train_list[i].replace('\n', '').split('\t')
        partition['train'].append([filename, start_idx])
    partition['validation'] = []
    for i in range(len(val_list)):
        label, filename, start_idx = val_list[i].replace('\n', '').split('\t')
        partition['validation'].append([filename, start_idx])
    partition['evaluation'] = []
    for i in range(len(test_list)):
        label, filename, start_idx = test_list[i].replace('\n', '').split('\t')
        partition['evaluation'].append([filename, start_idx])


    # Generators
    training_generator = DataGenerator(partition['train'], labels, cropsize, flag='random', **params)
    validation_generator = DataGenerator(partition['validation'], labels, cropsize, flag='fix', **params)
    evaluation_generator = DataGenerator(partition['evaluation'], labels, cropsize, flag='fix', **params)

    ## Load Model
    model = build_CRNN_200(input_shape)
    adam = optimizers.Adam(lr=0.001)
    model.compile(loss=amsoftmax_loss, optimizer=adam, metrics=['accuracy'])
    # Train model on dataset
    history = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  use_multiprocessing=True,
                                  epochs=epochs,
                                  verbose=1,
                                  callbacks=[early_stopping],
                                  workers=6)
    model.summary()
   
    now = datetime.datetime.now()
    Today = now.strftime('%Y%m%d')

    os.makedirs('best_model', exist_ok=True) 
    # Save model
    model.save('./best_model/model_speaker_2sec_64.h5')
    
    # Evaluation
    score = model.evaluate_generator(evaluation_generator)
    print('loss, acc : ', score)

    try:
        # loss graph
        fig, loss_ax = plt.subplots()
        loss_ax.plot(history.history['loss'], 'y', label='train_loss')
        loss_ax.plot(history.history['val_loss'], 'r', label='val_loss')
        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        plt.savefig('./best_model/training_curve.png')

        # loss graph
        fig, acc_ax = plt.subplots()
        acc_ax.plot(history.history['acc'], 'b', label='train_acc')
        acc_ax.plot(history.history['val_acc'], 'g', label='val_acc')
        acc_ax.set_xlabel('epoch')
        acc_ax.set_ylabel('acc')
        acc_ax.set_ylim([0, 1.1])
        fig.legend(loc='upper left')

        plt.savefig('./best_model/accuracy_curve.png')

    except Exception as e:
        print('Failed to save graph')
        print(e)
    clear_session()

if __name__ == '__main__':
    train_model()

