import cv2
import socket
import numpy as np

"""
Initialisiere die Socket-Verbindung
"""
def init_listen():
    host = ''
    port = 5555

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(10)
    # Bind the socket to the port
    server_address = (host, port)
    sock.bind(server_address)

    return sock

"""
Listen auf den Port 5555

Diese Funktion muss in einer MainLoop eingebaut werden damit ein Ständiges horchen auf dem Port möglich ist.
"""
def listen(sock):
    try:
        data, address = sock.recvfrom(65507)
    except socket.error as exc:
        exc = "%s" % exc
        if (exc == "timed out"):
            print("Timed Out")
            continue
        else:
            print("Caught exception socket.error in video thread: %s" % exc)
            break
    if(data is None):
        continue
    array = np.frombuffer(data, np.dtype('uint8'))
    data = cv2.imdecode(array, 1)

    return data
