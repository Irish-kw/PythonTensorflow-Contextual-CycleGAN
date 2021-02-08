from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from __init__ import glob_all
import matplotlib.pyplot as plt
import numpy as np
import os



def auto_delete_hdf5_save_history(history,path):
    file_list = glob_all(path)
    for i in range(len(file_list)-1):
        file_list.sort()
        if (len(file_list)-1) == 0:
            print('Only had Best Weight , Cound not delete !')
        else:
            os.remove(file_list[i])
            
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(path+'plt-loss.png')
    plt.clf()

def Encoder(channel_number):
    input_img = Input(shape=(348,204,1))
    e1 = Conv2D(16, (3, 3), activation='tanh', padding='same')(input_img)
    e2 = MaxPooling2D((2, 2), padding='same')(e1)
    e3 = Conv2D(8, (3, 3), activation='tanh', padding='same')(e2)
    e4 = MaxPooling2D((2, 2), padding='same')(e3)
    e5 = Conv2D(channel_number, (3, 3), activation='tanh', padding='same')(e4)
    e6 = MaxPooling2D((2, 2), padding='same')(e5)
    e7 = Activation('tanh')(e6)
    return Model(input_img, e7)

def Decoder(channel_number):
    input_img = Input(shape=(44,26,channel_number))
    d1 = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
    d2 = UpSampling2D((2, 2))(d1)
    d3 = Conv2D(16, (3, 3), activation='relu', padding='same')(d2)
    d4 = UpSampling2D((2, 2))(d3)
    d5 = Conv2D(32, (3, 3), activation='relu')(d4)
    d6 = UpSampling2D((2, 2))(d5)
    d7 = Conv2D(1, (3, 3), activation='relu', padding='same')(d6)
    return Model(input_img, d7)

def M1_1_AEModel(channel_number):
    input_img = Input(shape=(348,204,1))
    e1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    e2 = MaxPooling2D((2, 2), padding='same')(e1)
    e3 = Conv2D(8, (3, 3), activation='relu', padding='same')(e2)
    e4 = MaxPooling2D((2, 2), padding='same')(e3)
    e5 = Conv2D(channel_number, (3, 3), activation='relu', padding='same')(e4)
    e6 = MaxPooling2D((2, 2), padding='same')(e5)
    e7 = Activation('relu')(e6)
    d1 = Conv2D(8, (3, 3), activation='relu', padding='same')(e7)
    d2 = UpSampling2D((2, 2))(d1)
    d3 = Conv2D(16, (3, 3), activation='relu', padding='same')(d2)
    d4 = UpSampling2D((2, 2))(d3)
    d5 = Conv2D(32, (3, 3), activation='relu')(d4)
    d6 = UpSampling2D((2, 2))(d5)
    d7 = Conv2D(1, (3, 3), activation='tanh', padding='same')(d6)
    return Model(input_img, d7)

def M1_2_AEModel(channel_number):
    input_img = Input(shape=(348,204,1))
    e1 = Conv2D(16, (3, 3), activation='tanh', padding='same')(input_img)
    e2 = MaxPooling2D((2, 2), padding='same')(e1)
    e3 = Conv2D(8, (3, 3), activation='tanh', padding='same')(e2)
    e4 = MaxPooling2D((2, 2), padding='same')(e3)
    e5 = Conv2D(channel_number, (3, 3), activation='tanh', padding='same')(e4)
    e6 = MaxPooling2D((2, 2), padding='same')(e5)
    e7 = Activation('tanh')(e6)
    d1 = Conv2D(8, (3, 3), activation='tanh', padding='same')(e7)
    d2 = UpSampling2D((2, 2))(d1)
    d3 = Conv2D(16, (3, 3), activation='tanh', padding='same')(d2)
    d4 = UpSampling2D((2, 2))(d3)
    d5 = Conv2D(32, (3, 3), activation='tanh')(d4)
    d6 = UpSampling2D((2, 2))(d5)
    d7 = Conv2D(1, (3, 3), activation='tanh', padding='same')(d6)
    return Model(input_img, d7)

def M1_3_AEModel(channel_number):
    input_img = Input(shape=(348,204,1))
    e1 = Conv2D(16, (3, 3), activation='tanh', padding='same')(input_img)
    e2 = MaxPooling2D((2, 2), padding='same')(e1)
    e3 = Conv2D(8, (3, 3), activation='tanh', padding='same')(e2)
    e4 = MaxPooling2D((2, 2), padding='same')(e3)
    e5 = Conv2D(channel_number, (3, 3), activation='tanh', padding='same')(e4)
    e6 = MaxPooling2D((2, 2), padding='same')(e5)
    e7 = Activation('tanh')(e6)
    d1 = Conv2D(8, (3, 3), activation='relu', padding='same')(e7)
    d2 = UpSampling2D((2, 2))(d1)
    d3 = Conv2D(16, (3, 3), activation='relu', padding='same')(d2)
    d4 = UpSampling2D((2, 2))(d3)
    d5 = Conv2D(32, (3, 3), activation='relu')(d4)
    d6 = UpSampling2D((2, 2))(d5)
    d7 = Conv2D(1, (3, 3), activation='relu', padding='same')(d6)
    d8 = Dense(30, activation='tanh')(d7)
    d9 = Dense(10, activation='tanh')(d8)
    d10 = Dense(5, activation='tanh')(d9)
    d11 = Dense(1, activation='tanh')(d10)
    return Model(input_img, d11)