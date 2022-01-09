# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 14:06:21 2017

@author: dongming
"""

from keras.models import Sequential
from keras.layers import Conv3D, Input, merge, concatenate
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, adam
from keras.models import Model
import tensorflow as tf


from opts import *
import numpy as np
import os
import SimpleITK as sitk
import time


def model_unet(img_shape):
    concat_axis = 4
    inputs = Input(shape = img_shape)

    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv1)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv2)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)

    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv3)
    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)

    conv5 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
    conv5 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([conv5, conv4], axis=concat_axis)
    conv6 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([conv6, conv3], axis=concat_axis) 
    conv7 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([conv7, conv2], axis=concat_axis)
    conv8 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([conv8, conv1], axis=concat_axis)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv3D(1, (1, 1, 1))(conv9)
    model = Model(inputs=inputs, outputs=conv10)
       
    Adam = adam(lr=opt.learning_rate) #0.001
    model.compile(optimizer=Adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return model
  

def model_fcn():
    FCN = Sequential()
    FCN.add(BatchNormalization(input_shape=(None, None, None, 1)))
    FCN.add(Conv3D(64,3,3,3,init='he_normal', activation = 'relu',border_mode = 'same',
                     bias = True, input_shape = (None, None, None, 1)))
    
    FCN.add(Conv3D(32, 1, 1, 1, init = 'he_normal', activation = 'relu', border_mode = 'same', 
                     bias = True))
    FCN.add(Conv3D(32, 3, 3, 3, init='he_normal',activation='relu', border_mode='same', 
                     bias=True))
    FCN.add(Conv3D(32, 1, 1, 1, init = 'he_normal', activation = 'relu', border_mode = 'same', 
                     bias = True))
    FCN.add(Conv3D(16, 3, 3, 3, init='he_normal',activation='relu', border_mode='same', 
                     bias=True))
    FCN.add(Conv3D(1, 1, 1, 1, init='he_normal',
                     activation='linear', border_mode='valid', bias=True))

    Adam = adam(lr=0.001)
    FCN.compile(optimizer=Adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return FCN

def predict():
    Data_Path = opt.data_path # plase edit it 

    input_data = os.listdir(Data_Path)
    input_data = sorted(input_data)
    input_nums = input_data.__len__()
    input_nums = 2
    for i in range(input_nums/2):
        i = opt.subject_num-1 
        IMG_NAME = Data_Path + input_data[2*i]
        input_name = input_data[2*i]
        if opt.model == 'Unet':
            OUTPUT_NAME = input_name[:-4] + '_' + str(opt.model_num) + '_unet_smoothed.img'
        elif opt.model == 'FCN':
            OUTPUT_NAME = input_name[:-4] + '_fcn_smoothed.img'
        img = sitk.ReadImage(IMG_NAME)
        img_data = sitk.GetArrayFromImage(img)
        
	start = time.time()	

        shape = img_data.shape
        if opt.model == 'Unet':
            step = 16
        elif opt.model == 'FCN':
            step = 40
        img_test = np.zeros((shape[0],shape[1],shape[2]))

        for n in range(0,shape[2],step-8):
            
            startpoint = n
            
            if startpoint+step<=shape[2]:
                Y = np.zeros((1,shape[0],shape[1],step,1))
                Y[0,:,:,:,0] = img_data[:shape[0],:shape[1],startpoint:startpoint+step]
                if opt.model == 'Unet':
                    ms_net = model_unet([shape[0],shape[1],step,1])
                    
            else:
                Y = np.zeros((1,shape[0],shape[1],shape[2]-startpoint,1))
                Y[0,:,:,:,0] = img_data[:shape[0],:shape[1],startpoint:shape[2]]
                if opt.model == 'Unet':
                    ms_net = model_unet([shape[0],shape[1],shape[2]-startpoint,1])
            if opt.model == 'Unet':
                ms_net.load_weights("../model/m_model_adam30_3D_" + str(opt.model_num) + "_16_16.h5")
            
            if opt.model == 'FCN':
                ms_net = model_fcn()
                ms_net.load_weights("../model/m_model_adam30_3D.h5")
            pre = ms_net.predict(Y/255, batch_size=1)
            pre[pre[:] < 0] = 0
            
            if n==0:
                img_test[:,:,:startpoint + step -4] = pre[0,:,:,:-4,0]
            elif Y.shape[3] < step:
                img_test[:,:,startpoint+4:] = pre[0,:,:,4:,0]
            else:                     
                img_test[:shape[0],:shape[1],startpoint+4:startpoint+step-4] = pre[0,:,:,4:-4,0]

	end = time.time()	
        print(end-start)
        result = sitk.GetImageFromArray(img_test)
        result.CopyInformation(img)
        sitk.WriteImage(result, "../" + input_name[0:4] + "/" + OUTPUT_NAME)
        
        
        img_final = sitk.ReadImage("../" + input_name[0:4] + "/" + OUTPUT_NAME)
        img_float = sitk.GetArrayFromImage(img_final)
        img_float = img_float*255
        img_uint8 = img_float.astype(np.uint8)
        
        result = sitk.GetImageFromArray(img_uint8)
        result.CopyInformation(img)
        
        sitk.WriteImage(result, "../Level" + str(opt.model_num) + "/" + OUTPUT_NAME)
        del img_uint8,img_float,img_final,result,img_test,pre,Y,ms_net


if __name__ == "__main__":
    predict()

