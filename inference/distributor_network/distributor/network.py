import socket
import time
import numpy as np
import cv2

class network(object):
    def __init__(self, name, ip, port, com):
        self.name = name
        self.ip = ip
        self.port = int(port)
        self.com = com
        self.ping = 0

        self.start_time = 0
        self.end_time = 0

        self._setup()


    def __del__(self):
        self.sock.close()
        #self.send.close()

    # Erstelle eine Socket-Verbindung mit dem Timeout von 5 Sekunden
    def _setup(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(5.0)
        self.netCom = (self.ip, self.port)

        self.counter = 0

    """
    Hauptfunction

    - Sendet und Empfängt die Daten
    - Benutzt einen Timer um den Ping der Nachricht zu ermitteln.

    Bei dem Befehl Send, wird ebenfalls auf eine Nachricht gewartet. Dies soll
    als quittierung dienen und kann später zur Fehlervermeidung verwendet werden.
    """
    def run(self, img, com=None):
        data = img
        self.start_time = int(round(time.time() * 1000))
        try:

            # Sendet das Bild an ein Endgerät
            self._send(img, com)

            # Empfängt Daten
            data, server = self._listen()

            self.counter = 0
        except socket.error as e:
            self.counter += 1

        self.end_time = int(round(time.time() * 1000))
        self.ping  = (self.end_time - self.start_time)
        return self.counter, self.ping, data

    """
    Sollen Daten nur empfangen werden, wird der Befehl "get" gesendet, damit
    das Endgerät ein Bild an den Distributor sendet.
    bei den anderen Kommunikationensarten, wird ein Bild an das Endgerät gesendet.
    Dies aber auch nur wenn das Bild ungleich None ist.
    """
    def _send(self, img, com=None):
        if(self.com == "get"):
            sent = self.sock.sendto("get".encode('utf-8'), self.netCom)
        elif((self.com == "send") or (self.com == "both-UAV") or (self.com == "both-Dist")):
            #print(img is None)
            if(img is None):
                sent = self.sock.sendto(cv2_encode_image(np.zeros(shape=[512, 512, 3], dtype=np.uint8)), self.netCom)
            else:
                #print("Name: {}\n{}".format(com, img))
                # Wird benötigt um den Kommunikationsaufbau zu gewährleisten.
                if((com == "both-Dist")):
                    sent = self.sock.sendto(img, self.netCom)

                else:
                    sent = self.sock.sendto(cv2_encode_image(img), self.netCom)


    def _listen(self):
        img, server = self.sock.recvfrom(65507)
        return img, server

# Funktion zur verschlüsselung bzw. Komprimierung der Bilddaten
def cv2_encode_image(cv2_img, jpeg_quality=50):
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    result, buf = cv2.imencode('.jpg', cv2_img, encode_params)
    return buf.tobytes()
