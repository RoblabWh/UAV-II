from __future__ import print_function

import argparse
import os
import sys
import time
import scipy.io as sio
from PIL import Image

import tensorflow as tf
import numpy as np
import cv2

from model import DeepLabResNetModel

class Segmentation:



    def read_labelcolours(self, matfn):
        mat = sio.loadmat(matfn)
        color_table = mat['colors']
        shape = color_table.shape
        color_list = [tuple(color_table[i]) for i in range(shape[0])]

        return color_list

    def decode_labels(self, mask, num_images=1, num_classes=150):
        label_colours = self.read_labelcolours(self.matfn)

        n, h, w, c = mask.shape
        assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
        outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
        for i in range(num_images):
          img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
          pixels = img.load()
          for j_, j in enumerate(mask[i, :, :, 0]):
              for k_, k in enumerate(j):
                  if k < num_classes:
                      pixels[k_,j_] = label_colours[k]
          outputs[i] = np.array(img)
        return outputs

    def load(self, saver, sess, ckpt_path):
        saver.restore(sess, ckpt_path)
        print("Restored model parameters from {}".format(ckpt_path))

    def __init__(self):


        IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

        self.NUM_CLASSES = 27
        self.matfn = 'color150.mat'

        self.img = tf.placeholder("float", [None, None, 3])

        # Convert RGB to BGR.
        img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=self.img)
        self.img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
        # Extract mean.
        self.img -= IMG_MEAN

        # Create network.
        net = DeepLabResNetModel({'data': tf.expand_dims(self.img, dim=0)}, is_training=False, num_classes=self.NUM_CLASSES)

        # Which variables to load.
        restore_var = tf.global_variables()

        # Predictions.
        raw_output = net.layers['fc_out']
        raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(self.img)[0:2,])
        raw_output_up = tf.argmax(raw_output_up, dimension=3)
        self.pred = tf.expand_dims(raw_output_up, dim=3)

        # Set up TF session and initialize variables.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth = True)

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        init = tf.global_variables_initializer()

        self.sess.run(init)

        # Load weights.
        ckpt = tf.train.get_checkpoint_state("./restore_weights/ResNet101/")

        if ckpt and ckpt.model_checkpoint_path:
            loader = tf.train.Saver(var_list=restore_var)
            load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
            self.load(loader, self.sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found.')
            load_step = 0

        i = -1


    def get_segment(self, image):
        #start = time.clock()
        preds = self.sess.run(self.pred, feed_dict={self.img: image})
        #end = time.clock()
        #print(end - start)

        msk = self.decode_labels(preds, num_classes=self.NUM_CLASSES)
        frame = msk[0]
        return frame


