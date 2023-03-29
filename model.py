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
 if __name__=='__main__':
    train_deeplabv3plus_keras_xception_cbam()
