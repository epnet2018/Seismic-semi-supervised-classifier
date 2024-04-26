# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:22:22 2023
This script is used to evaluate the performance changes of the model 
under different labeled sample sizes

@author: Luozhao Jia
email:123@eqha.gov.cn
"""
import numpy as np
from keras import backend
from keras.utils.np_utils import to_categorical
from keras import backend
import h5py
import model_helper

if __name__ =="__main__":
    train_set=np.empty(0)
    test_set=np.empty(0)

    hd5_test=r'.\data\new_train_event2022_3to1_ys1212_aug_test1.h5'
    #Read data into memory
            
    with h5py.File(hd5_test, 'r') as hf:
        test_set = hf['earthquake'][:]
    
    # Set random seeds to ensure consistency of each learning
    seed = 7
    np.random.seed(seed)
     
    #data preparation
    

    np.random.shuffle(test_set)
        

    

    
    Z2=test_set[:,:18000]  
    Z3=test_set[:,18003]
    X_validation=Z2.reshape(test_set.shape[0],1,100,180)
    Y_validation=Z3.reshape(-1,1)
    

    Y_test_Flag=Y_validation.copy()
    #Evaluation process
    model_path='.\\experiment_2\\model\\'
    
    y_test=[0,0,0,0]
    Y_test_hot_Flag= to_categorical(Y_test_Flag, num_classes=3)  
    
    Model_Name_Array=['c_model_1500_num_317','c_model_3050_num_317','c_model_4576_num_317','c_model_6100_num_317','c_model_7626_num_317','c_model_9152_num_317','c_model_10676_num_317','c_model_12200_num_317','c_model_13726_num_317','c_model_15252_num_317']

    res_array=[]
    for model_name in Model_Name_Array:
        vgg_model=model_helper.get_model(model_name,model_path)
        X_validation=Z2.reshape(Z2.shape[0],100,180,1)
        y_test[2]=vgg_model.predict(X_validation)
        res_array.append(model_helper.model_evaluation(model_name,y_test[2],Y_test_Flag))

    print('Model Performance Evaluation Table')
    print(np.round(np.array(res_array)*100,2).T)
