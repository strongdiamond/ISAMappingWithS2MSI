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
