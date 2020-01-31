import numpy as np
import tensorflow as tf
import threading

class Controller(threading.Thread):
    def __init__(self, tmp_stream, tello_drone, cnn_navigation_path, cnn_object_detection_path, eta=30):
        threading.Thread.__init__(self)
        self.tello_drone = tello_drone
        self.eta = eta
        self.tmp_stream = tmp_stream
        self.cnn_navigation = tf.keras.models.load_model(cnn_navigation_path)
        #self.cnn_object_detection = tf.keras.models.load_model(cnn_object_detection_path)

    def run(self):
        while True:
            frame = self.tello_drone.read()
            #objects = self.cnn_object_detection.predict(frame)
            #ToDo: Object Detection should override the IMU Data
            control_data_propability = self.cnn_navigation.predict(np.expand_dims(frame, axis=0))
            left_propability = control_data_propability[0][0]
            right_propability = control_data_propability[0][2]

            self.tmp_stream.setLines(left_propability, control_data_propability[0][1], right_propability)
            propability = left_propability - right_propability
            yaw_angle = propability * self.eta * 10
            if yaw_angle > 0:
                #self.tello_drone.send_command("cw " + str(yaw_angle))
                print("cw " + str(yaw_angle))
            else:
                #self.tello_drone.send_command("ccw " + str(yaw_angle * (-1)))
                print("ccw " + str(yaw_angle * (-1)))

            #self.tello_drone.send_command("forward 20")