# -*- coding: utf-8 -*-
import os
import sys
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras.layers import Conv2D,BatchNormalization,Activation,UpSampling2D
from keras.layers import Concatenate,GlobalAveragePooling2D,Reshape
from keras.models import Model,load_model

sys.path.append('.')

from xception_ljt.xception2_8 import Xception
from attention_module import attach_attention_module,se_block
#***********************************************************************
def ASPP(tensor):
    '''atrous spatial pyramid pooling'''
    dims = K.int_shape(tensor)
    print(dims)
    y_pool=GlobalAveragePooling2D() (tensor)   
    y_pool=Reshape((1,1,dims[3]))(y_pool)
    print(K.int_shape(y_pool))
    y_pool = Conv2D(filters=256, kernel_size=1, padding='same',
                    kernel_initializer='he_normal', name='pool_1x1conv2d', use_bias=False)(y_pool)
    y_pool = BatchNormalization(name=f'bn_1')(y_pool)
    y_pool = Activation('relu', name=f'relu_1')(y_pool)

    y_pool = UpSampling2D(size=(dims[1],dims[2]),interpolation='bilinear')(y_pool)

    y_1 = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
                 kernel_initializer='he_normal', name='ASPP_conv2d_d1', use_bias=False)(tensor)
    y_1 = BatchNormalization(name=f'bn_2')(y_1)
    y_1 = Activation('relu', name=f'relu_2')(y_1)

    y_6 = Conv2D(filters=256, kernel_size=3, dilation_rate=6, padding='same',
                 kernel_initializer='he_normal', name='ASPP_conv2d_d6', use_bias=False)(tensor)
    y_6 = BatchNormalization(name=f'bn_3')(y_6)
    y_6 = Activation('relu', name=f'relu_3')(y_6)

    y_12 = Conv2D(filters=256, kernel_size=3, dilation_rate=12, padding='same',
                  kernel_initializer='he_normal', name='ASPP_conv2d_d12', use_bias=False)(tensor)
    y_12 = BatchNormalization(name=f'bn_4')(y_12)
    y_12 = Activation('relu', name=f'relu_4')(y_12)

    y_18 = Conv2D(filters=256, kernel_size=3, dilation_rate=18, padding='same',
                  kernel_initializer='he_normal', name='ASPP_conv2d_d18', use_bias=False)(tensor)
    y_18 = BatchNormalization(name=f'bn_5')(y_18)
    y_18 = Activation('relu', name=f'relu_5')(y_18)
   
    y = Concatenate(name='ASPP_concat')([y_pool, y_1, y_6, y_12, y_18])
    y = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
               kernel_initializer='he_normal', name='ASPP_conv2d_final', use_bias=False)(y)
    y = BatchNormalization(name=f'bn_final')(y)
    y = Activation('relu', name=f'relu_final')(y)
    return y
def DeepLabV3Plus(img_height=512, img_width=512,nchannels=12, nclasses=1):
    base_model = Xception(include_top=True,input_shape=(img_height,img_width,nchannels),weights=None)
    print(type(base_model.layers)) 
    print(len(base_model.layers))
    #*************************************************************************
    highFeatures=base_model.get_layer('block13_conv2_bn').output
    x_a=attach_attention_module(highFeatures,'cbam_block1')
    x_a = ASPP(x_a)
    x_a = UpSampling2D(size=(4,4),interpolation='bilinear')(x_a)
    x_a=se_block(x_a,'se_block_a')
    ##********************************************************
    lowerFeatures=base_model.get_layer('block3_conv2_bn').output
    x_b=attach_attention_module(lowerFeatures,'cbam_block2')
    x_b = Conv2D(filters=48, kernel_size=1, padding='same',
                 kernel_initializer='he_normal', name='low_level_projection', use_bias=False)(x_b)
    x_b = BatchNormalization(name=f'bn_low_level_projection')(x_b)
    x_b = Activation('relu', name='low_level_activation')(x_b)
    x_b=se_block(x_b,'se_block_b')
    #*************************************************************************
    x = Concatenate(name='decoder_concat')([x_a, x_b])
    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
               kernel_initializer='he_normal', name='decoder_conv2d_1', use_bias=False)(x)
    x = BatchNormalization(name=f'bn_decoder_1')(x)
    x = Activation('relu', name='activation_decoder_1')(x)
    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
               kernel_initializer='he_normal', name='decoder_conv2d_2', use_bias=False)(x)
    x = BatchNormalization(name=f'bn_decoder_2')(x)
    x = Activation('relu', name='activation_decoder_2')(x)
    
    x = UpSampling2D(size=(4,4),interpolation='bilinear')(x)
    
    x = Conv2D(nclasses, (1, 1), name='output_layer',padding="same")(x)
    
    model = Model(inputs=base_model.input, outputs=x, name='DeepLabV3_Plus')
    print(f'*** Output_Shape => {model.output_shape} ***')
    return model
#########################################################################
if __name__=='__main__':
    patch_h, patch_w, nchannels= 512, 512,12
    num_classes=1
    model = DeepLabV3Plus(patch_h, patch_w,nchannels, num_classes)
    plot_model(model,to_file='./graph/cbam.png',show_shapes=True)
    pass
