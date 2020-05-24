# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:12:13 2019

@author: truptiramdas.chavan
Aim: To obtain inference results on test set

Modify:
1. NUM_CLASSES,lam, kernel as per requirement
2. YOUR PATH TO TES IMAGEs: path to the test images
3. YOUR PATH TO MODEL WEIGHTS: path to the model trained weights
4. label1: names of labels
5. YOUR PATH TO SAVE INFERENCE IMAGES: path to save inference image
6. imgidx = any integer between 0-number of test images  #provide image index for which to predict

"""
import cv2
import os
import tensorflow as tf
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from ssd import SSD300
from ssd_utils import BBoxUtility

#define the preprocessing
def img_sharpen(img):
    sharp_img = np.zeros((img.shape))
    smooth_img = cv2.filter2D(img,-1,kernel)
    edge_img = img-smooth_img
    sharp_img = img+lam*edge_img
    return sharp_img

#define image path classes and other info
img_path = '8.jpeg'
NUM_CLASSES = 2
input_shape = (300, 300, 3)
lam = 1.5
kernel = np.ones((5,5),np.float32)/25
inputs = []
images = []  
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(cv2.imread(img_path))
inputs.append(img_sharpen(img.copy()))
inputs = np.array(inputs)#-127.0

bbox_util = BBoxUtility(NUM_CLASSES)

model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights('enhanced_SSD_weights.h5', by_name=True)

export_path='gs://dlaas-model/model/pb/knee/2'
#tf.saved_model.save(model, export_dir=export_path)


with tf.keras.backend.get_session() as sess:
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={'input_image': model.input},
        outputs={t.name: t for t in model.outputs})

