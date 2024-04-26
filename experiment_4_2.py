# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:22:22 2023
Testing of methods using the Kentucky induced earthquake dataset in the United States
@author: Luozhao Jia
email:123@eqha.gov.cn
"""
import numpy as np
from keras import backend
from keras.utils.np_utils import to_categorical
from keras import backend
import h5py
import model_helper


if __name__ == '__main__':

    model_path='.\\experiment_4\\model_save\\'
    
    hd5_test=r'.\\data\\temp_test.h5'
    with h5py.File(hd5_test, 'r') as hf:
        test_set = hf['earthquake'][:]
    seed = 7
    np.random.seed(seed)
    np.random.shuffle(test_set)
    Z2=test_set[:,:18000]  
    Z3=test_set[:,18003]
    
    X_validation=Z2.reshape(test_set.shape[0],100,180,1)

    Y_test_Flag=Z3.reshape(-1)
    y_test = []
    Y_test_hot_Flag= to_categorical(Y_test_Flag, num_classes=2)  
    Model_Name_Array=['c_model_200_Latent_sample30ktx','c_model_2280_num3533ktx']
    res_array=[]
    res_array1=[]
    for model_name in Model_Name_Array:

        vgg_model=model_helper.get_model(model_name,model_path)
        X_validation=Z2.reshape(Z2.shape[0],100,180,1)
        y_test=vgg_model.predict(X_validation)
        res_array.append(model_helper.model_evaluation(model_name,y_test,Y_test_Flag))
        res_array1.append(y_test)
    print('Model Performance Evaluation Table:\r\nAccuracy Precision F1 Recall')
    print(np.round(np.array(res_array)*100,2))
