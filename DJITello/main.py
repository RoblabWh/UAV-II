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
    eta = 10.0

    tmp_stream = videoStream.videostream(drone)

    #ToDO: In den Controller auslagern.
    drone.send_command("takeoff")
    drone.send_command("cw 0")
    time.sleep(7)
    drone.send_command("up 50")
    time.sleep(5)
    drone.send_command("speed 20")
    
    while True:
        try:
            #time.sleep(1)
            frame = cv2.cvtColor(cv2.resize(drone.read(), (101, 101)), cv2.COLOR_RGB2GRAY)/255.0
            # objects = self.cnn_object_detection.predict(frame)
            # ToDo: Object Detection should override the IMU Data
            # print(frame.shape)
            control_data_propability = cnn_navigation.predict(np.expand_dims(np.expand_dims(frame, axis=0), axis=3))
            left_propability = control_data_propability[0][0]
            right_propability = control_data_propability[0][2]

            tmp_stream.setLines(left_propability, control_data_propability[0][1], right_propability)
            propability = left_propability - right_propability
            yaw_angle = propability * eta
            #if yaw_angle > 0:
                #drone.send_command("cw " + str(int(yaw_angle)))
                #print("cw " + str(int(yaw_angle)))
            #else:
                #drone.send_command("ccw " + str(int(yaw_angle * (-1))))
                #print("ccw " + str(int(yaw_angle * (-1))))
            rc_drone_command = "rc 0 " + str(int(control_data_propability[0][1] * 40)) + " 0 " + str(int(propability * 40))
            drone.send_command(rc_drone_command)
            print(rc_drone_command)
            #drone.send_command("rc 0 0 0 20")
        except:
            print("Unexpected error:", sys.exc_info()[0])
            #raise

        #drone.send_command("forward 100")
        #time.sleep(1)
        drone.send_command("command")
        #print("forward 30")


if __name__ == "__main__":
    main()
