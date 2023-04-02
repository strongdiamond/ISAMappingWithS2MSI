# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import cv2
import tifffile as tiff
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from keras.models import Model,load_model
from keras.callbacks import  ModelCheckpoint,EarlyStopping,CSVLogger,ReduceLROnPlateau
from keras import backend as K
from keras_xception_cbam280 import DeepLabV3Plus
from category_focal_loss_for_ISA import BinaryFocalLoss
from ISASamplesProvider280_v3 import SparseSamplePatchBatch,ShuffleCallback,get_shuffle_img_gt_train_val_fns
###*******************************************************************************#####
patch_size=(512,512)
patch_h, patch_w, nchannels= 512, 512,12
img_size=512
num_classes=1
imgs_gts_path='/samples'
batch_size = 3
def train_deeplabv3plus_keras_xception_cbam():
    train_same_prefix_fnames,valid_same_prefix_fnames=get_shuffle_img_gt_train_val_fns(imgs_gts_path,val_samples_num=450)
    train_gen = SparseSamplePatchBatch(train_same_prefix_fnames,imgs_gts_path,batch_size, patch_size)
    val_gen = SparseSamplePatchBatch(valid_same_prefix_fnames,imgs_gts_path,batch_size, patch_size)
    train_steps = train_gen.__len__()
    K.clear_session()
    weights_dir = '/demo/Infor/deeplab_cbam_ISA_weights/weights'
    checkpoint_dir='/demo/Infor/deeplab_cbam_ISA_weights/ckpt'
    csv_dir='/demo/Infor/deeplab_cbam_ISA_weights/csv'
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    
    best_weights_filepath=os.path.join(weights_dir,'isa_v1.h5') 
    model=DeepLabV3Plus(patch_h, patch_w,nchannels, num_classes)
    if os.path.exists(best_weights_filepath):
        model.load_weights(best_weights_filepath)
        print('load weights==>'+best_weights_filepath)
        best_weights_filepath=os.path.join(weights_dir,'isa_v2.h5')
    else:
        pass    
    
    checkpoint_cb = ModelCheckpoint(best_weights_filepath,monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    earlystopping_cb=EarlyStopping(monitor='val_loss',patience=10,verbose=1,mode='auto',restore_best_weights=True)
    csv_logger = CSVLogger(csv_dir+'/isa_v1.csv', append=True, separator=',')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=1,min_lr=0.001)
               
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                    loss=BinaryFocalLoss(alpha=0.25,gamma=2.0),metrics=['accuracy'])
   
    callbacks =[checkpoint_cb,reduce_lr,earlystopping_cb,csv_logger]
    epochs =50
    
    model.fit(train_gen, validation_data=val_gen,epochs=epochs,initial_epoch=0,steps_per_epoch=train_steps,callbacks=callbacks)
 if __name__=='__main__':
    train_deeplabv3plus_keras_xception_cbam()
