import time
import socket
import sys
import cv2
import numpy as np


class Distributer:
    def __init__(self, ip_uav, port_uav, ip_dist, port_dist):
        self.ip_uav = ip_uav
        self.port_uav = port_uav
        self.ip_dist = ip_dist
        self.port_dist = port_dist
        self.listen = None
        self.send = None

    def __del__(self):
        self.listen.close
        self.send.close
        print('destructor called')

    def run(self):
        self.listen = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        uav_network = (self.ip_uav, self.port_uav)

        self.send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        dist_network = (self.ip_dist, self.port_dist)
        self.send.bind(dist_network)
        #data_sent, address = send.recvfrom(4)
        #sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
        #img, server = listen.recvfrom(65507)
        #listen.setblocking(0)
        # = listen.sendto("get".encode('utf-8'), uav_network
        #a = listen.settimeout(10.0)
        #print("vor listen")
        #img, server = listen.recvfrom(65507)

        #print("listen")
        #listen.setblocking(0)
        while(True):
            try:
                start_time = int(round(time.time() * 1000))
                sent = self.listen.sendto("get".encode('utf-8'), uav_network)
                #print("vor listen")
                img, server = self.listen.recvfrom(65507)
                self.listen.settimeout(1.0)
                #print("listen")
                #listen.setblocking(0)
                array = np.frombuffer(img, dtype=np.dtype('uint8'))
                image = cv2.imdecode(array, 1)
                cv2.imshow("test", image)

                #test = 'Test Dominik'
                data, address = self.send.recvfrom(60000)
                data = data.decode('utf-8')
                self.send.sendto(img, address)
                #print("gesendet")

                end_time = int(round(time.time() * 1000))
                sec = int(round((end_time - start_time) / 1000))
                ms = (end_time - start_time) % 1000
                sys.stdout.write('\r Listen and Send in {}s and {} ms'.format(int(round((end_time - start_time) / 1000)), (end_time - start_time) % 1000))
                sys.stdout.flush()
                #while(ms < 100):
                    #end_time = int(round(time.time() * 1000))
                    #ms = (end_time - start_time) % 1000

                #time.sleep(1)
                if ((cv2.waitKey(1) & 0xFF == ord('q'))):
                    self.listen.sendto("quit".encode('utf-8'), uav_network)
                    break
            except socket.error as e:
                continue
            except:
                break

def main():
    uav_ip = '172.16.35.169'
    uav_port = 5555
    dist_ip = ''
    dist_port = 5555

    dist = Distributer(uav_ip, uav_port, dist_ip, dist_port)
    dist.run()



if __name__== "__main__":
    main()
