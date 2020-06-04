# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 12:22:24 2020
​
@author: Marc
"""

import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
filepath = r'/media/gas/Samsung_T5/kinectKellerProcessed/kinectTransP4/pack8/rgb2011816595_963303_rgbd2011816595_952209/202011816595.963303_projected.pgm'

imgN = np.expand_dims(np.expand_dims(np.array(cv2.imread(filepath, 0)),0), 3)
#imgN = np.array(cv2.imread(filepath))
print(imgN.shape)

images_placeholder = tf.placeholder(tf.float32, shape=[1,240, 320,1])
gradImgX,gradImgY = tf.image.image_gradients(images_placeholder)

with tf.Session() as sess:
    # Run every operation with variable input
    print( "Addition with variables: %i" % sess.run(gradImgX, feed_dict={images_placeholder: imgN}))


print(np.min(gradImgX), np.max(gradImgX), np.mean(gradImgX))
gradImgShowX = gradImgX + np.abs(np.min(gradImgX))
gradImgShowX = gradImgShowX * (255 / np.max(gradImgX))
gradImgShowX = gradImgShowX.astype(np.uint8)


cv2.imshow(gradImgShowX)
cv2.waitKey()