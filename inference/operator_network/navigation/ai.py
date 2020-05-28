#!/usr/bin/env python

"""ai.py: controlls the drone via a cnn"""

__author__ = "Artur Leinweber"
__copyright__ = "Copyright 2020, UAV-II Project"
__credits__ = ["Gerhard Senkowski", "Dominik Slomma"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Artur Leinweber"
__email__ = "artur.leinweber@studmail.w-hs.de"
__status__ = "Production"

import sys
import time

import tensorflow as tf
import numpy as np
import cv2

import drone


class ai:
    LEFT = 0
    FORWARD = 1
    RIGHT = 2
    DIMENSION = 0


    def __init__(self, drone):
        self.cnn_orientation = tf.keras.models.load_model("./gray.h5")
        self.theta = 8
        self.scaling = 40
        self.running = True
        self.drone = drone


    def stop(self):
        self.running = False


    def run(self):
        self.drone.send_command("moff")
        time.sleep(2)
        self.drone.send_command("cw 0")


        time.sleep(7)
        self.drone.send_command("up 50")
        time.sleep(5)
        self.drone.send_command("speed 20")
        i = 0

        while self.running:
            try:
                frame = cv2.cvtColor(cv2.resize(drone.getFrame(), (101, 101)), cv2.COLOR_RGB2GRAY) / 255.0

                control_data_propability_orientation = self.cnn_orientation.predict(np.expand_dims(np.expand_dims(frame, axis=0), axis=3))

                left_propability_orientation = control_data_propability_orientation[DIMENSION][LEFT]
                right_propability_orientation = control_data_propability_orientation[DIMENSION][RIGHT]
                forward_propability_orientation = control_data_propability_orientation[DIMENSION][FORWARD]

                i += 1
                if i == self.theta:
                    propability_orientation = left_propability_orientation - right_propability_orientation
                    rc_drone_command = "rc 0 " + str(int(forward_propability_orientation * self.scaling)) + " 0 " + str(

                        int(propability_orientation * self.scaling))
                    self.drone.send_command(rc_drone_command)
                    i = 0
            except:
                print("Unexpected error:", sys.exc_info()[0])
                self.drone.send_command("land")

            drone.send_command("command")