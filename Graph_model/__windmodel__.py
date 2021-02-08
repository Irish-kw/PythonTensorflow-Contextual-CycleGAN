from random import random
from numpy.random import randint
from numpy import zeros,ones,asarray
from RandomTesting.__RandomTesting__ import Random_Testing,calculate_extract_loss
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import losses
from tensorflow.keras import backend as K
from tensorflow import keras
import math
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow.compat.v1 as tf
import glob
import os
import numpy as np
import pandas as pd
import warnings
import gc
import time
warnings.filterwarnings('ignore')
import logging
logging.getLogger().disabled = True

class WindLossLayer(Layer):
    def __init__(self, **kwargs):
        self.radius = 3
        self.surrounding_pixel = 1
        self.pi_direction = K.constant(180,dtype='float32')
        self.pi_const = K.constant(math.pi,dtype='float32')
        self.alpha = 0.2
        
        self.sta_lat=[9,61,66,70,70,74,74,76,77,81,81,82,87,89,89,
                 102,112,118,118,125,144,160,160,169,185,185,
                 189,189,189,205,206,210,211,220,223,227,229,
                 230,236,239,252,270,277,283,288,288,294,300,
                 304,309,309,311,312,313,315,315,316,317,317,
                 318,319,320,320,320,320,320,321,321,324,326,
                 330,331,332]

        self.sta_lon=[79,42,57,34,43,31,32,36,29,30,49,34,33,31,116,
                 54,21,22,117,30,32,25,45,35,21,55,26,35,68,69,
                 41,97,160,55,68,47,65,62,57,75,76,83,180,90,109,
                 175,98,122,104,121,123,154,146,131,146,152,153,
                 109,144,151,158,121,149,152,153,165,137,150,152,
                 177,145,169,153]

        self.x_lst = [1,0,0,0,1,2,2,2]
        self.y_lst = [2,2,1,0,0,0,1,2]
        self.idx_lst = [0,1,2,3,4,5,6,7]
        
        super(WindLossLayer, self).__init__(**kwargs)

    def my_tf_round(self, x, decimals = 0):
        multiplier = tf.constant(10**decimals, dtype=x.dtype)
        return (tf.round(x * multiplier) / multiplier)

    def all_coordinate(self, x, y):
        self.all_coordinate_lst = []
        for j in range(self.radius):
            for i in range(self.radius):
                self.all_coordinate_lst.append([x+(i - (self.surrounding_pixel)), 
                                                y+(j - (self.surrounding_pixel))])
        return(tf.convert_to_tensor(self.all_coordinate_lst))        
        
    def wind_loss(self, x, y, y_pred, EPA73_WindDir, EPA73_WindSpd):
        whole_coordinate = self.all_coordinate(x, y)

        local_matrix = y_pred[0,whole_coordinate[0][0]:whole_coordinate[-1][0]+1,
                              whole_coordinate[0][1]:whole_coordinate[-1][1]+1,0]


        local_matrix_diff = []
        for local_i in range(self.radius):
            for local_j in range(self.radius):
                local_matrix_diff.append(local_matrix[local_i,local_j] - 
                                         local_matrix[self.surrounding_pixel,self.surrounding_pixel])

        local_matrix_diff = K.reshape(local_matrix_diff,(self.radius,self.radius))

        local_matrix_diff_notation = self.my_tf_round((local_matrix_diff) / (K.abs(local_matrix_diff+1e-5)),2)
        local_matrix_diff_exp_abs = K.exp(K.abs(local_matrix_diff))

        local_matrix_diff_exp_abs_norm = local_matrix_diff_exp_abs / (K.sum(local_matrix_diff_exp_abs)-1)
        local_matrix_diff_exp_abs_norm_sign = (local_matrix_diff_exp_abs_norm * local_matrix_diff_notation)

        WindDir_cos = K.cos((EPA73_WindDir[0,x,y,0]/self.pi_direction) * self.pi_const)
        WindDir_sin = K.sin((EPA73_WindDir[0,x,y,0]/self.pi_direction) * self.pi_const)

        self.d_coefficient = []
        for _ in range(8):self.d_coefficient.append((WindDir_cos * (K.cos((self.pi_const * _)/4))) + 
                                                    (WindDir_sin * (K.sin((self.pi_const * _)/4))))

        self.d_sum = 0
        for x_coor, y_coor, idx in (zip(self.x_lst, self.y_lst, self.idx_lst)):
            self.d_sum += (local_matrix_diff_exp_abs_norm_sign[x_coor,y_coor] * self.d_coefficient[idx])

        similarity = ((self.d_sum+1)/2)
        penality = EPA73_WindSpd[0,x,y,0] * (1-similarity)
        
        return (penality)        

    def call(self, inputs):
        station_loss = 0
        y_pred, EPA73_WindDirec, EPA73_WindSpeed = inputs
        
        for lat,lon in zip(self.sta_lat, self.sta_lon):
            station_loss += self.wind_loss(lat, lon, y_pred, EPA73_WindDirec, EPA73_WindSpeed)
        self.add_loss((self.alpha*(station_loss/73)), inputs=inputs)
        return (y_pred)

def glob_all(dir_path):
    file_list = glob.glob(os.path.join(dir_path,'*.h5'))
    inside = os.listdir(dir_path)
    for dir_name in inside:
        if os.path.isdir(os.path.join(dir_path,dir_name)):
            file_list.extend(glob_all(os.path.join(dir_path,dir_name)))
    return file_list

def real_samples(dataset, train_direc, train_windspd, n_samples, patch_shape1, patch_shape2):
    ix = randint(0, dataset.shape[0], n_samples)
    iy = randint(0, train_direc.shape[0], n_samples)
    x = dataset[ix]
    x1 = train_direc[iy]
    x2 = train_windspd[iy]
    y = ones((n_samples, patch_shape1, patch_shape2, 1))
    return x, x1, x2, y

def fake_samples(g_model, dataset, patch_shape1, patch_shape2):
    x = g_model.predict(dataset)
    y = zeros((len(x), patch_shape1, patch_shape2, 1))
    return x, y

def resnet_block(n_filters, input_layer):
    init = RandomNormal(stddev=0.02)
    g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
    g = tfa.layers.InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
    g = tfa.layers.InstanceNormalization(axis=-1)(g)
    g = Concatenate(axis=3)([g, input_layer])
    return g

def define_discriminator(image_shape, d_lr):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    in_image = Input(shape=image_shape)
    # C64
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = tfa.layers.InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = tfa.layers.InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = tfa.layers.InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = tfa.layers.InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(patch_out)
    # define model
    model = Model(in_image, patch_out)
    # compile model
    model.compile(loss='mse', optimizer=tf.keras.optimizers.RMSprop(lr=d_lr), 
                  loss_weights=[0.5],metrics=['accuracy'])
    return model

def define_generator(image_shape, n_resnet=9):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # c7s1-64
    g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
    g = tfa.layers.InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # d128
    g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = tfa.layers.InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # d256
    g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = tfa.layers.InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # R256
    for _ in range(n_resnet):
        g = resnet_block(256, g)
    # u128
    g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = tfa.layers.InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # u64
    g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = tfa.layers.InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # c7s1-3
    g = Conv2D(1, (7,7), padding='same', kernel_initializer=init)(g)
    g = tfa.layers.InstanceNormalization(axis=-1)(g)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image)
    return model

def define_composite_model(g_model_1, d_model, g_model_2, image_shape, g_lr=5e-5):

    # ensure the model we're updating is trainable
    g_model_1.trainable = True
    # mark discriminator as not trainable
    d_model.trainable = False
    # mark other generator model as not trainable
    g_model_2.trainable = False
    
    # discriminator element
    input_gen = Input(shape=image_shape)
    gen1_out = g_model_1(input_gen)
    output_d = d_model(gen1_out)
    
    # identity element
    input_id = Input(shape=image_shape)
    output_id = g_model_1(input_id)
    
    # forward cycle
    output_f = g_model_2(gen1_out)
    
    # backward cycle
    gen2_out = g_model_2(input_id)
    output_b = g_model_1(gen2_out)
    # define model graph

    model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])

    # define optimization algorithm configuration
    opt = tf.keras.optimizers.RMSprop(lr=g_lr)
    # compile model with weighting of least squares loss and L1 loss
    model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
    return model

def WindLossmodel():
    y_pred_pm25 = Input(shape=(348,204,1))
    epa_winddirec = Input(shape=(348,204,1))
    epa_windspeed = Input(shape=(348,204,1))

    x_out = WindLossLayer()([y_pred_pm25, epa_winddirec, epa_windspeed])
    return Model(([y_pred_pm25,epa_winddirec,epa_windspeed]), x_out)

def define_composite_wind_model(g_model_AtoB, WindLoss_model, image_shape, g_lr=5e-5):

    g_model_AtoB.trainable = True
    input_gen = Input(shape=(348,204,1))
    input_direc = Input(shape=(348,204,1))
    input_windspd = Input(shape=(348,204,1))

    B_map = g_model_AtoB(input_gen)
    B_map = WindLoss_model([B_map, input_direc, input_windspd])
    
    model = Model([input_gen, input_direc, input_windspd], B_map)
    opt = tf.keras.optimizers.RMSprop(lr=g_lr)
    model.compile(loss=None, optimizer=opt)
    return model

def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, c_model_wind,
          dataA, dataB, model_path, EPA73_test_pm25, train_direc, train_windspd,
          station_coordinate, ex5_lst, ex10_lst, ex15_lst, ex20_lst, ex20_KNN_norm, ex20_KNNhalf, 
          n_epochs, n_batch, save_epochs):
    
    n_epochs, n_batch, = n_epochs, n_batch
    n_patch1 = d_model_A.output_shape[1]
    n_patch2 = d_model_A.output_shape[2]
    trainA, trainB  = dataA, dataB
    poolA, poolB = list(), list()
    bat_per_epo = int(len(trainA) / n_batch)
    n_steps = bat_per_epo * n_epochs
    print(bat_per_epo)
 

    for i in range(n_steps):
        X_realA, train_direc, train_windspd, y_realA = real_samples(trainA, train_direc, train_windspd, 
                                                                    n_batch, n_patch1, n_patch2) #348X204
        
        X_realB, _, _, y_realB = real_samples(trainB, train_direc, train_windspd, n_batch, n_patch1, n_patch2) #348X204
        
        X_fakeA, y_fakeA = fake_samples(g_model_BtoA, X_realB, n_patch1, n_patch2)
        X_fakeB, y_fakeB = fake_samples(g_model_AtoB, X_realA, n_patch1, n_patch2)
        
        g_loss2, _, _, _, _ = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])

        dA_loss1, dA_acc1 = d_model_A.train_on_batch(X_realA, y_realA)
        dA_loss2, dA_acc2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)

        g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])

        dB_loss1,dB_acc1 = d_model_B.train_on_batch(X_realB, y_realB)
        dB_loss2,dB_acc2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
        
#         wind_loss1 = c_model_wind.train_on_batch([X_realA,train_direc,train_windspd])
        wind_loss2 = c_model_wind.train_on_batch([X_fakeA,train_direc,train_windspd])
        
        print('>%05d, dALoss[%.3f,%.3f] dAAcc[%.3f,%.3f] dBLoss[%.3f,%.3f] dBAcc[%.3f,%.3f] g[%.3f,%.3f] wind[%.3f]' % 
              (i+1, dA_loss1, dA_loss2, dA_acc1, dA_acc2, dB_loss1, dB_loss2, dB_acc1, dB_acc2, g_loss1, g_loss2, wind_loss2))

        if (i+1) % (bat_per_epo * save_epochs) == 0:
            save_models(i, g_model_AtoB, g_model_BtoA, d_model_A, d_model_B, model_path) 
            tStart = time.time()
            print('Start testing!')

            ex_mse20_list, ex_mae20_list, ex_mape20_list = ([] for _it in range(3))

            for iter_test in range(30):
                a ,lats_ex20,lons_ex20=Random_Testing(4, iter_test, station_coordinate, EPA73_test_pm25, ex5_lst, ex10_lst, 
                                                      ex15_lst, ex20_lst)
                SAT_pre_matrix20 = g_model_AtoB.predict(ex20_KNN_norm[iter_test])
                SAT_pre_matrix20 = ((SAT_pre_matrix20 * ex20_KNNhalf[iter_test]) + ex20_KNNhalf[iter_test])


                mse20, mae20 ,mape20 = calculate_extract_loss(SAT_pre_matrix20, EPA73_test_pm25, lats_ex20, lons_ex20)
                del SAT_pre_matrix20

                print('>%05d, iter_test%02d, mse20=%05d' % (i+1, iter_test, mse20))

                ex_mse20_list.append(mse20); ex_mae20_list.append(mae20); ex_mape20_list.append(mape20)                 

            tEnd = time.time()
            print ("It cost %f sec" % (tEnd - tStart))

            f4 = open(model_path+'result/ex_mse20_list.txt', 'a')
            print('----------'+str(i+1)+'----------',file=f4)
            print('max:  '+str(np.array(ex_mse20_list).max()),file=f4)
            print('min:  '+str(np.array(ex_mse20_list).min()),file=f4)
            print('mean: '+str(np.array(ex_mse20_list).mean()),file=f4)
            print('var:  '+str(np.array(ex_mse20_list).var()),file=f4)
            print('std:  '+str(np.array(ex_mse20_list).std()),file=f4)
            print('---------------------------',file=f4)             
            f4.close()                 

            f4 = open(model_path+'result/ex_mae20_list.txt', 'a')
            print('----------'+str(i+1)+'----------',file=f4)
            print('max:  '+str(np.array(ex_mae20_list).max()),file=f4)
            print('min:  '+str(np.array(ex_mae20_list).min()),file=f4)
            print('mean: '+str(np.array(ex_mae20_list).mean()),file=f4)
            print('var:  '+str(np.array(ex_mae20_list).var()),file=f4)
            print('std:  '+str(np.array(ex_mae20_list).std()),file=f4)
            print('---------------------------',file=f4)             
            f4.close()

            f4 = open(model_path+'result/ex_mape20_list.txt', 'a')
            print('----------'+str(i+1)+'----------',file=f4)
            print('max:  '+str(np.array(ex_mape20_list).max()),file=f4)
            print('min:  '+str(np.array(ex_mape20_list).min()),file=f4)
            print('mean: '+str(np.array(ex_mape20_list).mean()),file=f4)
            print('var:  '+str(np.array(ex_mape20_list).var()),file=f4)
            print('std:  '+str(np.array(ex_mape20_list).std()),file=f4)
            print('---------------------------',file=f4)             
            f4.close()                

            gc.collect()

    return (None)


def save_models(step, g_model_AtoB, g_model_BtoA, d_model_A, d_model_B, model_path):
    filename1 = model_path+'/g_model_AtoB_%07d.h5' % (step+1)
    g_model_AtoB.save(filename1)
    filename2 = model_path+'/g_model_BtoA_%07d.h5' % (step+1)
    g_model_BtoA.save(filename2) 
    filename3 = model_path+'/d_model_A_%07d.h5' % (step+1)
    d_model_A.save(filename3)
    filename4 = model_path+'/d_model_B_%07d.h5' % (step+1)
    d_model_B.save(filename4)    
    print('>Saved: %s and %s' % (filename1[filename1.find('g_model'):], filename2[filename2.find('g_model'):]))