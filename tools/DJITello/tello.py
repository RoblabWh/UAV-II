import socket
import threading
import time
import numpy as np
import libh264decoder
from logger import write_to_log
import cv2 as cv

class Tello:
    # Wrapper Klasse zum interagieren mit der Tello Drohne

    def __init__(self, local_ip, local_port, imperial=False, command_timeout=.3, tello_ip='192.168.10.1',
                 tello_port=8889):
        write_to_log("Call the constructor from the tello")
        self.decoder = libh264decoder.H264Decoder()
        self.imperial = imperial
        self.frame = None  # numpy array BGR -- current camera output frame
        self.cmd_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # socket for sending cmd
        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # socket for receiving video stream
        self.tello_address = (tello_ip, tello_port)
        self.tello_ip = tello_ip
        self.local_video_port = 11111  # port for receiving video stream
        self.last_height = 0
        self.cmd_socket.bind((local_ip, local_port))


        # to receive video -- send cmd: command, streamon
        self.cmd_socket.sendto(b'command', self.tello_address)
        write_to_log("Send 'command' to initialise the SDK")
        self.cmd_socket.sendto(b'streamon', self.tello_address)
        write_to_log("Send 'streamon' to start the videostream")


        self.video_socket.bind((local_ip, self.local_video_port))


        # thread for receiving video
        write_to_log("Start a thread for a videostream")
        self.receive_video_thread = threading.Thread(target=self._receive_video_thread)
        self.receive_video_thread.daemon = True
        self.receive_video_thread.start()


    # Destruktor
    def __del__(self):
        write_to_log("Call the destructor from tello.py")
        self.cmd_socket.close()
        self.video_socket.close()

    def _receive_video_thread(self):
        """
        Listens for video streaming (raw h264) from the Tello.
        Runs as a thread, sets self.frame to the most recent frame Tello captured.
        """
        write_to_log("Videostream thread started")
        packet_data = ""
        while True:
            try:
                res_string, ip = self.video_socket.recvfrom(2048)
                packet_data += res_string
                # end of frame
                if len(res_string) != 1460:
                    for frame in self._h264_decode(packet_data):
                        # the frame is a numpy array BGR, it must be converted
                        b, g, r = cv.split(frame)
                        frame = cv.merge((r,g,b))
                        self.frame = frame
                    packet_data = ""

            except socket.error as exc:
                print ("Caught exception socket.error : %s" % exc)

    def _h264_decode(self, packet_data):
        """
        decode raw h264 format data from Tello
        :param packet_data: raw h264 data array
        :return: a list of decoded frame
        """
        res_frame_list = []
        frames = self.decoder.decode(packet_data)
        for framedata in frames:
            (frame, w, h, ls) = framedata
            if frame is not None:
                # print 'frame size %i bytes, w %i, h %i, linesize %i' % (len(frame), w, h, ls)

                frame = np.fromstring(frame, dtype=np.ubyte, count=len(frame), sep='')
                frame = (frame.reshape((h, ls / 3, 3)))
                frame = frame[:, :w, :]
                res_frame_list.append(frame)

        return res_frame_list

    def read(self):
        """Return the last frame from camera."""
        return self.frame

    def send_command(self, command):
        """
        Send a command to the Tello.
        :param command: (str) the command to send
        """
        self.cmd_socket.sendto(command.encode('utf-8'), self.tello_address)
        write_to_log('sending command: %s to %s' % (command, self.tello_ip))
