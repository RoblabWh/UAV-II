import cv2
import socket
import numpy as np

"""
Bitte verwenden!!!

Die Kameradaten der Drohne sind zunächst falsch kodiert (BGR) und müssen in GBR gewandelt werden.
"""
def convBGRtoRGB(frame):
    b, g, r = cv2.split(frame)
    frame = cv2.merge((r, g, b))
    return frame

""" Funktion zum Komprimieren von Bildern """
def cv2_encode_image(cv2_img, jpeg_quality=50):
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    result, buf = cv2.imencode('.jpg', cv2_img, encode_params)
    return buf.tobytes()

"""
Funktion der Drohne

Diese Funktion wird von der Drohne aufgerufen. Sie lauscht auf dem Port 5555 und wartet darauf, dass eine Verbindung von Seiten des Distributor erstellt wird.
"""
def video_server(drone):
    host = ''
    port = 5555
    clients = []


    #global keep_running
    keep_running = not drone.terminate

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(10)
    # Bind the socket to the port
    server_address = (host, port)

    #print('starting up on %s port %s\n' % server_address)

    sock.bind(server_address)


    # Lauf solange, bis der Operator abgeschaltet wird.
    while(keep_running):
        keep_running = not drone.terminate
        try:
            data, address = sock.recvfrom(65507)
        except socket.error as exc:
            exc = "%s" % exc
            if(exc == "timed out"):
                #print("test erfolgreich")
                continue
            else:
                print ("Caught exception socket.error in video thread: %s" % exc)
                break
        #img = data
        #print("Data: {} Address: {}".format(data, address))
        #enc = data.decode('utf-8')
        array = np.frombuffer(data, np.dtype('uint8'))
        data = cv2.imdecode(array, 1)
        print("decode")
        print(data)
        if((data == "get") or (data is None)):
            #for i in range(0, len(clients))

            buffer = cv2_encode_image(drone.getFrame())
            if buffer is None:
                continue
            if len(buffer) > 65507:
                print("The message is too large to be sent within a single UDP datagram. We do not handle splitting the message in multiple datagrams")
                sock.sendto("FAIL".encode('utf-8'), address)
                continue
            # We send back the buffer to the client
            sock.sendto(buffer, address)
        else:
            print("else zweig")
            if(data is None):
                continue
            print("set image")
            drone.setManipulatedFrame(data)
            buffer = cv2_encode_image(drone.getFrame())
            if buffer is None:
                continue
            if len(buffer) > 65507:
                print(
                    "The message is too large to be sent within a single UDP datagram. We do not handle splitting the message in multiple datagrams")
                sock.sendto("FAIL".encode('utf-8'), address)
                continue
            # We send back the buffer to the client
            sock.sendto(buffer, address)
        """data, address = sock.recvfrom(4)
        img = data.copy()
        print("Data: {} Address: {}".format(data, address))
        data = data.decode('utf-8')
        if (data == "get"):
            for i in range(0, len(clients))

            buffer = cv2_encode_image(drone.getFrame())
            if buffer is None:
                continue
            if len(buffer) > 65507:
                print(
                    "The message is too large to be sent within a single UDP datagram. We do not handle splitting the message in multiple datagrams")
                sock.sendto("FAIL".encode('utf-8'), address)
                continue
            # We send back the buffer to the client
            sock.sendto(buffer, address)

        for client in clients:
            if(address == client):
                array = np.frombuffer(img, dtype=np.dtype('uint8'))
                image = cv2.imdecode(array, 1)
                cv2.imshow("test", image)
                if (cv2.waitKey(1) & 0xFF == ord('q')):
                    #listen.sendto("quit".encode('utf-8'), server_adress)
                    break
    cv2.destroyAllWindows()"""


            # We send back the buffer to the client
            #sock.sendto(buffer, address)

        # elif(data == "quit"):
        #     grabber.stop()
        #     keep_running = False

    print("Network establish Quitting..")
    sock.close()
