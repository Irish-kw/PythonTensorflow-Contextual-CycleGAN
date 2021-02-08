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


def glob_all(dir_path):
    file_list = glob.glob(os.path.join(dir_path,'*.h5'))
    inside = os.listdir(dir_path)
    for dir_name in inside:
        if os.path.isdir(os.path.join(dir_path,dir_name)):
            file_list.extend(glob_all(os.path.join(dir_path,dir_name)))
    return file_list

def real_samples(dataset, n_samples, patch_shape1, patch_shape2):
    ix = randint(0, dataset.shape[0], n_samples)
    x = dataset[ix]
    y = ones((n_samples, patch_shape1, patch_shape2, 1))
    return x, y

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

def define_composite_model_A2B(g_model_1, d_model, g_model_2, image_shape, g_lr=5e-5):

    # ensure the model we're updating is trainable
    g_model_1.layers[1].trainable = True
    g_model_1.layers[2].trainable = False
    g_model_1.layers[3].trainable = True
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

def define_composite_model_B2A(g_model_1, d_model, g_model_2, image_shape, g_lr=5e-5):

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


def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, 
          dataA, dataB, model_path, EPA73_test_pm25, station_coordinate, ex5_lst, ex10_lst, 
          ex15_lst, ex20_lst, ex20_KNN_norm, ex20_KNNhalf, n_epochs, n_batch, save_epochs):
    
    n_epochs, n_batch, = n_epochs, n_batch
    n_patch1 = d_model_A.output_shape[1]
    n_patch2 = d_model_A.output_shape[2]
    trainA, trainB  = dataA, dataB
    poolA, poolB = list(), list()
    bat_per_epo = int(len(trainA) / n_batch)
    n_steps = bat_per_epo * n_epochs
    print(bat_per_epo)
 

    for i in range(n_steps):
        X_realA, y_realA = real_samples(trainA, n_batch, n_patch1, n_patch2) #348X204
        X_realB, y_realB = real_samples(trainB, n_batch, n_patch1, n_patch2) #348X204
        
        X_fakeA, y_fakeA = fake_samples(g_model_BtoA, X_realB, n_patch1, n_patch2)
        X_fakeB, y_fakeB = fake_samples(g_model_AtoB, X_realA, n_patch1, n_patch2)
        
        g_loss2, _, _, _, _ = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])

        dA_loss1, dA_acc1 = d_model_A.train_on_batch(X_realA, y_realA)
        dA_loss2, dA_acc2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)

        g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])

        dB_loss1,dB_acc1 = d_model_B.train_on_batch(X_realB, y_realB)
        dB_loss2,dB_acc2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)

        print('>%05d, dALoss[%.3f,%.3f] dAAcc[%.3f,%.3f] dBLoss[%.3f,%.3f] dBAcc[%.3f,%.3f] g[%.3f,%.3f]' % 
              (i+1, dA_loss1, dA_loss2, dA_acc1, dA_acc2, dB_loss1, dB_loss2, dB_acc1, dB_acc2, g_loss1, g_loss2))

        if (i+1) % (bat_per_epo * save_epochs) == 0: 
            save_models(i, g_model_AtoB, g_model_BtoA, d_model_A, d_model_B, model_path) 
            tStart = time.time()
            print('Start testing!')

            ex_mse20_list, ex_mae20_list, ex_mape20_list = ([] for _it in range(3))

            for iter_test in range(30):
                a ,lats_ex20,lons_ex20=Random_Testing(4, iter_test, station_coordinate, EPA73_test_pm25,
                                                      ex5_lst, ex10_lst, ex15_lst, ex20_lst)
                SAT_pre_matrix20 = g_model_AtoB.predict(ex20_KNN_norm[iter_test])
                SAT_pre_matrix20 = ((SAT_pre_matrix20 * ex20_KNNhalf[iter_test]) + ex20_KNNhalf[iter_test])

                mse20, mae20 ,mape20 = calculate_extract_loss(SAT_pre_matrix20, EPA73_test_pm25, lats_ex20,
                                                              lons_ex20)
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