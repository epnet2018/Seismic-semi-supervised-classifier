# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:09:16 2023

@author: Luozhao Jia
email: 123@eqha.gov.cn
This page's code is used to assist in loading pre-trained models 
and evaluating the performance of the models, supporting evaluation 
of accuracy, precision,f1,recall.

"""

import numpy as np
from datetime import datetime,timedelta  
from pandas import Series,DataFrame
import pandas as pd
from math import radians, cos, sin, asin, sqrt  
import threading
import itertools
import os
import tensorflow as tf
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, Activation
from keras.layers import Flatten
from keras.layers.convolutional import  Conv2D
from keras.layers.convolutional import  Conv1D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import GlobalAveragePooling1D
from keras.utils import np_utils
from keras import backend
from keras.utils.np_utils import to_categorical
from sklearn import metrics
from keras import backend
from keras.utils import plot_model
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.initializers import RandomNormal
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.models import model_from_json
import h5py
backend.set_image_data_format('channels_first')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def get_model(model_name,model_path):
    '''
    This function is to read the model into memory
    
    Parameters
    ----------
    model_name : model name
    model_path : model path

    Returns
    -------
    Model variables
    '''
    #model path
    model_file='%s%s.h5'%(model_path,model_name)
    new_model = load_model(model_file)
    #Create a new optimizer
    optimizer = keras.optimizers.Adam(learning_rate=0.001,beta_1=0.5)
    new_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return new_model


def model_evaluation(model_name,y_test,y_true):
    '''
    This function is used to evaluate the model

    Parameters
    ----------
    model_name : model name
    wave_array : evaluation data

    Returns
    -------
    model_acc : accuracy
    model_p : precision
    model_f1 : f1
    model_r : recall

    '''
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score, precision_score, recall_score
    from sklearn.metrics import balanced_accuracy_score
    y_pred=np.argmax(y_test,axis=1).reshape(-1,1)
    model_acc=accuracy_score(y_true, y_pred)
    model_f1 = f1_score( y_true, y_pred, average='macro')
    model_p = precision_score(y_true, y_pred, average='macro')
    model_r = recall_score(y_true, y_pred, average='macro')
    model_b = balanced_accuracy_score(y_true, y_pred)
    result='modelname:{} \naccuracy:{:.3%},precision:{:.3%},f1:{:.3%},recall:{:.3%},balance:{:.3%}'.format(model_name,model_acc,model_p,model_f1, model_r,model_b)
    print(result)
    return model_acc,model_p,model_f1, model_r