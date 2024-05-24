# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:22:22 2023
This script uses a semi-supervised method to train data samples 
to form a new model.

@author: Luozhao Jia
email:123@eqha.gov.cn
"""
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
import tensorflow as tf
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import Activation
from matplotlib import pyplot
from keras import backend
import time
import matplotlib.pyplot as plt
from keras.utils import plot_model
import os
import numpy as np
import h5py
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
# Hyperparameter settings
img_dim_w=100
img_dim_h=180
sample_num=15252
layer_num=48
latent_dim = 150
# custom activation function
def custom_activation(output):
    logexpsum = backend.sum(backend.exp(output), axis=-1, keepdims=True)
    result = logexpsum / (logexpsum + 1.0)
    return result
# define the standalone supervised and unsupervised discriminator models
def define_discriminator(in_shape=(img_dim_w,img_dim_h,1), n_classes=3):
    in_image = Input(shape=in_shape)
    fe = Conv2D(layer_num, (3,3), strides=(2,2), padding='same')(in_image)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv2D(layer_num, (3,3), strides=(2,2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv2D(layer_num, (3,3), strides=(2,2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Flatten()(fe)
    fe = Dropout(0.4)(fe)
    fe = Dense(n_classes)(fe)
    c_out_layer = Activation('softmax')(fe)
    c_model = Model(in_image, c_out_layer)
    c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
    d_out_layer = Lambda(custom_activation)(fe)
    d_model = Model(in_image, d_out_layer)
    d_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    return d_model, c_model
# define the standalone generator model
def define_generator(latent_dim):
    in_lat = Input(shape=(latent_dim,))
    n_nodes = 16 * img_dim_w * img_dim_h
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((img_dim_w, img_dim_h, 16))(gen)
    gen = Conv2DTranspose(layer_num, (4,4), strides=(2,2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Conv2DTranspose(layer_num, (4,4), strides=(2,2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    out_layer = Conv2D(1, (img_dim_w,img_dim_h),strides=(4,4), activation='tanh', padding='same')(gen)
    model = Model(in_lat, out_layer)
    return model
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    d_model.trainable = False
    gan_output = d_model(g_model.output)
    model = Model(g_model.input, gan_output)
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model
# load the data
def load_real_samples():
    train_file=r'./data/new_train_event2022_3to1_ys1212_aug_train.h5'
    train_set=np.empty(0)
    validation_set=np.empty(0)
    with h5py.File(train_file, 'r') as hf:
        train_set = hf['earthquake'][:]
    train_set=train_set[np.where(train_set[:,18003]<3)]
    Z=train_set[:,:18000]  
    Z1=train_set[:,18003]
    X_train=Z.reshape(Z.shape[0],100,180,1)
    Y_train=Z1.reshape(-1)
    return [X_train, Y_train]
# select a supervised subset of the dataset, ensures classes are balanced
def select_supervised_samples(dataset, n_samples=sample_num, n_classes=3):
    X, y = dataset
    n_samples=X.shape[0]
    X_list, y_list = list(), list()
    n_per_class = int(n_samples / n_classes)
    for i in range(n_classes):
        X_with_class = X[y == i]
        if(X_with_class.shape[0]>n_per_class):
            ix =np.random.choice(len(X_with_class),n_per_class, replace=False)
        else:
            ix =np.random.choice(len(X_with_class),len(X_with_class), replace=False)
        [X_list.append(X_with_class[j]) for j in ix]
        [y_list.append(i) for j in ix]
    return asarray(X_list), asarray(y_list)

# select real samples
def generate_real_samples(dataset, n_samples):
    images, labels = dataset
    ix = randint(0, images.shape[0], n_samples)
    X, labels = images[ix], labels[ix]
    y = ones((n_samples, 1))
    return [X, labels], y
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    z_input = randn(latent_dim * n_samples)
    z_input = z_input.reshape(n_samples, latent_dim)
    return z_input
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    z_input = generate_latent_points(latent_dim, n_samples)
    images = generator.predict(z_input)
    y = zeros((n_samples, 1))
    return images, y
# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, c_model, latent_dim, dataset, n_samples=sample_num):
    X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    X = (X + 1) / 2.0
    X1=X.reshape(100,-1)
    path = r'.//model_save//'
    X, y = dataset
    _, acc = c_model.evaluate(X, y, verbose=0)
    print('Classifier Accuracy: %.3f%%' % (acc * 100))
    result='Classifier Accuracy: %.3f%%/n/n' % (acc * 100)
    filename =r'.//model_save//c_model_%s_result.txt'%latent_dim
    with open(filename, 'a') as file_object:
        file_object.write(result)
    filename3 =path+'c_model_%04d.h5' % (step+1)
    c_model.save(filename3)
    print('>Saved: %s' % (filename3))
# train the generator and discriminator
def train(g_model, d_model, c_model, gan_model, dataset, latent_dim, n_epochs=20, n_batch=30):
    X_sup, y_sup = select_supervised_samples(dataset)
    print(X_sup.shape, y_sup.shape)
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    n_steps = bat_per_epo * n_epochs
    half_batch = int(n_batch / 2)
    print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (n_epochs, n_batch, half_batch, bat_per_epo, n_steps))
    for i in range(n_steps):
        [Xsup_real, ysup_real], _ = generate_real_samples([X_sup, y_sup], half_batch)
        c_loss, c_acc = c_model.train_on_batch(Xsup_real, ysup_real)
        [X_real, _], y_real = generate_real_samples(dataset, half_batch)
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)
        X_gan, y_gan = generate_latent_points(latent_dim, n_batch), ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        if (i+1) % (bat_per_epo * 1) == 0:
            summarize_performance(i, g_model, c_model, latent_dim, dataset)
        print('>%d,c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (i+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss))
        if(np.isnan(c_loss)|np.isnan(d_loss1)|np.isnan(d_loss2)|np.isnan(g_loss)):
            summarize_performance(i, g_model, c_model, latent_dim, dataset)
            print('>>>>>%s>>>>>loss stop' %str(i+1))
            break
if __name__ == '__main__':
    tf.keras.backend.clear_session()
    d_model, c_model = define_discriminator()
    g_model = define_generator(latent_dim)
    gan_model = define_gan(g_model, d_model)
    dataset = load_real_samples()
    p1=time.time()
    train(g_model, d_model, c_model, gan_model, dataset, latent_dim)
    p2=time.time()
    filemodel =r'.//model_save//c_model_%s_Latent.h5' % (latent_dim)
    c_model.save(filemodel)
    print('train time: {:.2f} h'.format((p2-p1)/3600))