import socket
import numpy as np
import threading
#import cv2 as cv
import libh264decoder

#from utils import *
#from protocol import *

class drone(object):
    def __init__(self):
        #Init Variables for the getter
        self.pitch = None
        self.roll = None
        self.yaw = None
        self.vgx = None
        self.vgy = None
        self.vgz = None
        self.h = None
        self.bat = None
        self.agx = None
        self.agy = None
        self.agz = None
        self.baro = None
        self.wifi = None
        #self.joystick = False

        #self.periodic = False
        #self.isSend = False
        self.cmd_response = None
        self.frame = None  # numpy array BGR -- current camera output frame
        self.abort_flag = False
        self.command_timeout = 0.3


        #terminate varibale
        self.terminate = False
        """
                    Prepare send and receive Commands
        """
        self.cmd_socket = self.initSocket('', 8889, 5) # maybe Port 9000
        self.receive_cmd = threading.Thread(target=self._receive_cmd_thread)
        self.tello_cmd_address = ('192.168.10.1', 8889)
        # self.tello_cmd_address = ('192.168.2.120', 8889)
        #self.abort_flag = False
        """
                    Prepare Receive tello-state
        """
        self.socket_state = self.initSocket('', 8890)
        self.receive_socket_state = threading.Thread(target=self._receive_state_thread)
        """
                    Prepare Receive Videostream
        """
        self.video_socket = self.initSocket('', 11111)
        self.receive_video_thread = threading.Thread(target=self._receive_video_thread)
        self.receive_video_thread.daemon = True
        self.decoder = libh264decoder.H264Decoder()


        """
                    Init SDK Mode & start Streaming
        """
        self.cmd_socket.sendto(b'command', self.tello_cmd_address)
        self.cmd_socket.sendto(b'streamon', self.tello_cmd_address)
        #self.periodicTimer = threading.Timer(10, self.send_periodically)
        """
                    Start the threads
        """
        self.receive_cmd.start()
        self.receive_socket_state.start()
        self.receive_video_thread.start()

    def __del__(self):
        self.cmd_socket.close()
        self.socket_state.close()
        self.video_socket.close()
        self.setTerminate()

    def _receive_cmd_thread(self):
        """
        Listen to responses from the Tello.

        Runs as a thread, sets self.response to whatever the Tello last returned.

        """
        while True:
            if self.terminate:
                break
            try:
                #print(self.joystick)
                """if(self.joystick):
                    #print("recv_cmd Thread")
                    left_x, left_y, right_y, right_x = self._normalize()
                    command = "rc {} {} {} {}".format(left_x, left_y, right_y, right_x)
                    print(left_x, left_y, right_y, right_x)
                    self.cmd_socket.sendto(command.encode('utf-8'), self.tello_cmd_address)
                    #self._send_stick_cmd()
                    response, ip = self.cmd_socket.recvfrom(2000)
                else:"""
                response, ip = self.cmd_socket.recvfrom(2000)
                self.cmd_response = response
                #print(response)
            except socket.error as exc:
                pass
                #print("Caught exception socket.error in cmd thread: %s" % exc)
        print("command thread offline")



    def _receive_state_thread (self):
        while True:
            if self.terminate:
                break
            try:
                #print("state thread start")
                response, ip = self.socket_state.recvfrom(3000)
                #print("response state trhread: " + response)
                value = []
                description = []
                dic = {}
                sp = response.split(";")
                #print ("deklaration set")
                for i in range(0, 21):
                    counts = sp[i].split(":")
                    # print(counts[1])
                    description.append((counts[0]))
                    value.append(counts[1])
                    dic[description[i]] = value[i]
                    #print("in for")
                #print("out for")
                #print (dic['h'])
                self.pitch = dic['pitch']
                self.roll = dic['roll']
                self.yaw = dic['yaw']
                self.vgx = dic['vgx']
                self.vgy = dic['vgy']
                self.vgz = dic['vgz']
                self.h = dic['h']
                self.bat = dic['bat']
                self.agx = dic['agx']
                self.agy = dic['agy']
                self.agz = dic['agz']
                self.baro = dic['baro']
                #print("Response rdy")
                # test = "mid:-1;x:0;y:0;z:0;mpry:0,0,0;pitch:5;roll:13;yaw:-84;vgx:0;vgy:0;vgz:0;templ:73;temph:75;tof:10;h:0;bat:73;baro:68.86;time:0;agx:104.00;agy:-237.00;agz:-978.00;"
                # print("Ausgabe: {}".format(test))
                #print(dic)
                #print (response)
                #mid:-1;x:0;y:0;z:0;mpry:0,0,0;pitch:5;roll:13;yaw:-84;vgx:0;vgy:0;vgz:0;templ:73;temph:75;tof:10;h:0;bat:73;baro:68.86;time:0;agx:104.00;agy:-237.00;agz:-978.00;
                #if(self.mode not 'normal'):
                #write_data(response)
                #print(response)
                #print(str(ip))
                #self.battery = response[ response.index(";bat:") + 5 : response.index(";baro:")]
                #print("Batterielevel: {}".format(self.battery))
                #self.height = response[ response.index(";h:") + 3 : response.index(";bat:")]
                #print("Hoehe: {}".format(self.height))
            except socket.error as exc:
                print ("Caught exception socket.error in state thread : %s" % exc)
        print("state thread offline")

    def _receive_video_thread(self):
        """
        Listens for video streaming (raw h264) from the Tello.

        Runs as a thread, sets self.frame to the most recent frame Tello captured.

        """
        packet_data = ""
        #print("start videostream")
        while True:
            if self.terminate:
                break
            try:
                res_string, ip = self.video_socket.recvfrom(3000)
                packet_data += res_string
                # end of frame
                if len(res_string) != 1460:
                    #cv.imshow('TelloStream', packet_data)
                    for frame in self._h264_decode(packet_data):
                        # the frame is a numpy array BGR, it must be converted
                        #b, g, r = cv.split(frame)
                        #frame = cv.merge((r,g,b))
                        self.frame = frame
                        #cv.imshow('TelloStream', frame)
                    packet_data = ""
            except socket.error as exc:
                print ("Caught exception socket.error in video thread: %s" % exc)
        #print("video thread offline")

    def initSocket(self, adress, port, timeout=None):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # socket for receiving video stream
        sock.bind((adress, port))
        if(not(timeout == None)):
            sock.settimeout(timeout)
        return sock
    """
           Decode the Videostream frame
       """
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

    """
            Send a command to the tello Drone
        """
    """def send_periodically(self):
        print("periodicly")
        self.periodicTimer.cancel()
        timer = threading.Timer(self.command_timeout, self.set_abort_flag)
        self.periodic = True
        while self.isSend:
            pass
        # mal mit command versuchen!!
        self.cmd_socket.sendto("wifi?".encode('utf-8'), self.tello_cmd_address)
        timer.start()
        while self.cmd_response is None:
            if self.abort_flag is True:
                break
        timer.cancel()

        if self.cmd_response is None:
            response = 'none_response'
        else:
            self.wifi = self.cmd_response.decode('utf-8')

        print(self.wifi)
        self.cmd_response = None
        self.periodic = False
        self.periodicTimer = threading.Timer(10, self.send_periodically)
        self.periodicTimer.start()
    """
    def send_command(self, command):
        """
        Send a command to the Tello and wait for a response.

        :param command: Command to send.
        :return (str): Response from Tello.

        """

        #print(">> send cmd: {}".format(command))
        self.abort_flag = False

        timer = threading.Timer(self.command_timeout, self.set_abort_flag)

        #while (self.periodic):
            #pass
        #self.isSend = True
        self.cmd_socket.sendto(command.encode('utf-8'), self.tello_cmd_address)

        timer.start()
        while self.cmd_response is None:
            if self.abort_flag is True:
                break
        timer.cancel()

        if self.cmd_response is None:
            response = 'none_response'
        else:
            response = self.cmd_response.decode('utf-8')

        self.cmd_response = None
        #self.isSend = False
        return response

    """
           Abort-Flag
       """

    def set_abort_flag(self):
        """
        Sets self.abort_flag to True.

        Used by the timer in Tello.send_command() to indicate to that a response

        timeout has occurred.

        """

        self.abort_flag = True

    """
        Command Befehle
    """
    def cmdTakeoff(self):
        return self.send_command('takeoff')

    def cmdLand(self):
        return self.send_command('land')

    def cmdUp(self, x):
        return self.send_command('up %s' % x)

    def cmdDown(self, x):
        return self.send_command('down %s' % x)

    def cmdLeft(self, x):
        return self.send_command('left %s' % x)

    def cmdRight(self, x):
        return self.send_command('right %s' % x)

    def cmdForward(self, x):
        return self.send_command('forward %s' % x)

    def cmdBack(self, x):
        return self.send_command('back %s' % x)

    def cmdCw(self, x):
        return self.send_command('cw %s' % x)

    def cmdCcw(self, x):
        return self.send_command('ccw %s' % x)

    def cmdRC(self, left_x, left_y, right_x, right_y):
        #left_x, left_y, right_y, right_x = self._normalize()
        command = "rc {} {} {} {}".format(left_x, left_y, right_x, right_y*-1)

        print(left_x, left_y, right_x, right_y)
        #self.send_command("rc {} {} {} {}".format(left_x, left_y, right_y, right_x))
        self.cmd_socket.sendto(command.encode('utf-8'), self.tello_cmd_address)

    """
        Setz Befehle fuer die Drohne
    """
    def setTerminate(self):
        self.terminate = True

    def setSpeed(self, x):
        pass

    def set_joystick(self, stick):
        if(stick):
            self.cmd_socket.settimeout(0.2)
            #self.command_timeout = 0
        else:
            self.cmd_socket.settimeout(5)
        #self.joystick = stick

    def rc_counter_clockwise(self, val):
        """
        CounterClockwise tells the drone to rotate in a counter-clockwise direction.
        Pass in an int from 0-100.
        """
        command = "ccw {}".format(val)
        self.cmd_socket.sendto(command.encode('utf-8'), self.tello_cmd_address)

    def rc_clockwise(self, val):
        """
        Clockwise tells the drone to rotate in a clockwise direction.
        Pass in an int from 0-100.
        """
        command = "cw {}".format(val)
        self.cmd_socket.sendto(command.encode('utf-8'), self.tello_cmd_address)

    def rc_left(self, val):
        """Left tells the drone to go left. Pass in an int from 0-100."""
        command = "left {}".format(val)
        self.cmd_socket.sendto(command.encode('utf-8'), self.tello_cmd_address)

    def rc_right(self, val):
        """Right tells the drone to go right. Pass in an int from 0-100."""
        command = "right {}".format(val)
        self.cmd_socket.sendto(command.encode('utf-8'), self.tello_cmd_address)

    def rc_backward(self, val):
        """Backward tells the drone to go in reverse. Pass in an int from 0-100."""
        command = "back {}".format(val)
        self.cmd_socket.sendto(command.encode('utf-8'), self.tello_cmd_address)

    def rc_forward(self, val):
        """Forward tells the drone to go forward. Pass in an int from 0-100."""
        command = "forward {}".format(val)
        self.cmd_socket.sendto(command.encode('utf-8'), self.tello_cmd_address)

    def rc_down(self, val):
        """Down tells the drone to descend. Pass in an int from 0-100."""
        command = "down {}".format(val)
        self.cmd_socket.sendto(command.encode('utf-8'), self.tello_cmd_address)

    def rc_up(self, val):
        """Up tells the drone to ascend. Pass in an int from 0-100."""
        command = "up {}".format(val)
        self.cmd_socket.sendto(command.encode('utf-8'), self.tello_cmd_address)

    """
        Getter
    """
    def getFrame(self):
        return self.frame

    def getWifi(self):
        return self.wifi

    def getPitch(self):
        return self.pitch

    def getRoll(self):
        return self.roll

    def getYaw(self):
        return self.yaw

    def getVgx(self):
        return self.vgx

    def getVgy(self):
        return self.vgy

    def getVgz(self):
        return self.vgz

    def getH(self):
        return self.h

    def getBat(self):
        return self.bat

    def getAgx(self):
        return self.agx

    def getAgy(self):
        return  self.agy

    def getAgz(self):
        return self.agz

    def getBaro(self):
        return self.baro