import tello
import videoStream
from controller import Controller
from logger import write_to_log
import time
import numpy as np
import tensorflow as tf
import threading
import cv2
import sys

def main():
    write_to_log("Initialise Tello Drone")
    drone = tello.Tello('', 8889)
    cnn_navigation = tf.keras.models.load_model("./model.h5")
    eta = 30.0

    tmp_stream = videoStream.videostream(drone)

    #ToDO: In den Controller auslagern.
    drone.send_command("takeoff")
    while True:
        try:
            frame = cv2.resize(drone.read(), (101, 101))
            # objects = self.cnn_object_detection.predict(frame)
            # ToDo: Object Detection should override the IMU Data
            control_data_propability = cnn_navigation.predict(np.expand_dims(frame, axis=0))
            left_propability = control_data_propability[0][0]
            right_propability = control_data_propability[0][2]

            tmp_stream.setLines(left_propability, control_data_propability[0][1], right_propability)
            propability = left_propability - right_propability
            yaw_angle = propability * eta * 10
            if yaw_angle > 0:
                drone.send_command("cw " + str(int(yaw_angle)))
                print("cw " + str(int(yaw_angle)))
            else:
                drone.send_command("ccw " + str(int(yaw_angle * (-1))))
                print("ccw " + str(int(yaw_angle * (-1))))
        except:
            print("Unexpected error:", sys.exc_info()[0])
            #raise

        # self.tello_drone.send_command("forward 20")


if __name__ == "__main__":
    main()
