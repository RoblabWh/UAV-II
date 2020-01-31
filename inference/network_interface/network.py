import cv2
import socket
import numpy as np
<<<<<<< HEAD

=======
""" Bitte verwenden!!! """
def convBGRtoRGB(frame):
    b, g, r = cv2.split(frame)
    frame = cv2.merge((r, g, b))
    return frame
>>>>>>> efc9209e61adb2a087c45dd47e1d081d12e288f6

def cv2_encode_image(cv2_img, jpeg_quality=50):
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    result, buf = cv2.imencode('.jpg', cv2_img, encode_params)
    return buf.tobytes()

def video_server(drone):
    host = ''
    port = 5555
    clients = []


    keep_running = True

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind the socket to the port
    server_address = (host, port)

    print('starting up on %s port %s\n' % server_address)

    sock.bind(server_address)

    while(keep_running):
        data, address = sock.recvfrom(4)
        img = data
        print("Data: {} Address: {}".format(data, address))
        data = data.decode('utf-8')
        if(data == "get"):
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

    print("Quitting..")
    sock.close()
