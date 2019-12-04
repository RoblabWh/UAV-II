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
import socket

def cv2_encode_image(cv2_img, jpeg_quality=50):
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    result, buf = cv2.imencode('.jpg', cv2_img, encode_params)
    return buf.tobytes()

def video_server(drone):
    host         = ''
    port         = 5555

    keep_running = True

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind the socket to the port
    server_address = (host, port)

    print('starting up on %s port %s\n' % server_address)

    sock.bind(server_address)

    while(keep_running):
        data, address = sock.recvfrom(4)
        data = data.decode('utf-8')
        if(data == "get"):
            buffer = cv2_encode_image(drone.read())
            if buffer is None:
                continue
            if len(buffer) > 65507:
                print("The message is too large to be sent within a single UDP datagram. We do not handle splitting the message in multiple datagrams")
                sock.sendto("FAIL".encode('utf-8'), address)
                continue

            # We send back the buffer to the client
            sock.sendto(buffer, address)

        # elif(data == "quit"):
        #     grabber.stop()
        #     keep_running = False

    print("Quitting..")
    sock.close()

def main():
    write_to_log("Initialise Tello Drone")
    drone = tello.Tello('', 8889)
    
    video_server_thread = threading.Thread(target=video_server, args=(drone,))
    
    
    cnn_orientation = tf.keras.models.load_model("./gray.h5")
    cnn_offset = tf.keras.models.load_model("./offset_small_51.h5")
    eta = 10.0
    theta = 8
    
    tmp_stream = videoStream.videostream(drone)

    #ToDO: In den Controller auslagern.
    drone.send_command("takeoff")
    drone.send_command("cw 0")
    time.sleep(7)
    drone.send_command("up 50")
    time.sleep(5)
    drone.send_command("speed 20")
    i = 0
    forward_propability_orientation = 0.0
    left_propability_orientation = 0.0
    right_propability_orientation = 0.0
    forward_propability_offset = 0.0
    left_propability_offset = 0.0
    right_propability_offset = 0.0
    video_server_thread.start()
    while True:
        try:
            #before = time.time()
            #time.sleep(1)
            frame = cv2.cvtColor(cv2.resize(drone.read(), (101, 101)), cv2.COLOR_RGB2GRAY)/255.0
            # objects = self.cnn_object_detection.predict(frame)
            # ToDo: Object Detection should override the IMU Data
            # print(frame.shape)
            control_data_propability_orientation = cnn_orientation.predict(np.expand_dims(np.expand_dims(frame, axis=0), axis=3))
            control_data_propability_offset = cnn_offset.predict(np.expand_dims(np.expand_dims(frame, axis=0), axis=3))
            left_propability_orientation = control_data_propability_orientation[0][0]
            right_propability_orientation = control_data_propability_orientation[0][2]
            forward_propability_orientation = control_data_propability_orientation[0][1]
            left_propability_offset = control_data_propability_offset[0][0]
            right_propability_offset = control_data_propability_offset[0][2]
            forward_propability_offset = control_data_propability_offset[0][1]
            i += 1
            
            #yaw_angle = propability * eta
            #if yaw_angle > 0:
                #drone.send_command("cw " + str(int(yaw_angle)))
                #print("cw " + str(int(yaw_angle)))
            #else:
                #drone.send_command("ccw " + str(int(yaw_angle * (-1))))
                #print("ccw " + str(int(yaw_angle * (-1))))
            if i == theta:
                #forward_propability_orientation /= theta
                #left_propability_orientation /= theta
                #right_propability_orientation /= theta
                #forward_propability_offset /= theta
                #left_propability_offset /= theta
                #right_propability_offset /= theta
                tmp_stream.setLines(left_propability_orientation, control_data_propability_orientation[0][1], right_propability_orientation)
                propability_offset = left_propability_offset - right_propability_offset
                propability_orientation = left_propability_orientation - right_propability_orientation
                
                #rc_drone_command = "rc " + str(int(propability_offset * 40)) + " " + str(int(forward_propability_offset * 40)) + " 0  0"
                rc_drone_command = "rc 0 " + str(int(forward_propability_orientation * 40)) + " 0 " + str(int(propability_orientation * 40)) #orientaion
                print('forward_propability_orientation', forward_propability_orientation)
                #if forward_propability_orientation > 0.9:
                #    rc_drone_command = "rc " + str(int(propability_offset * 40)) + " " + str(int(forward_propability_offset * 40)) + " 0  0"
                #else:
                #    rc_drone_command = "rc 0 " + str(int(forward_propability_orientation * 40)) + " 0 " + str(int(propability_orientation * 40))
                    
                drone.send_command(rc_drone_command)
                print(rc_drone_command)
                i = 0
                left_propability_orientation = 0.0
                right_propability_orientation = 0.0
                forward_propability_orientation = 0.0
                left_propability_offset = 0.0
                right_propability_offset = 0.0
                forward_propability_offset = 0.0
            #drone.send_command("rc 0 0 0 20")
            #after = time.time()
            #print(str(after - before) + 'sekunden')
        except:
            print("Unexpected error:", sys.exc_info()[0])
            #raise 

        #drone.send_command("forward 100")
        #time.sleep(1)
        drone.send_command("command")
        #print("forward 30")
        
        


if __name__ == "__main__":
    main()
