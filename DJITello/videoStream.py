
import threading
import datetime
import cv2 as cv
import os
import time
import numpy as np

class videostream:

    def __init__(self,tello):


        self.tello = tello # videostream device

        self.frame = None  # frame read from h264decoder
        self.thread = None # thread of the OpenCV mainloop
        self.close = False
        self.overlay = np.full((100,90,3),255, np.uint8)
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()

    def __del__(self):
        """Closes the local socket."""
        cv.destroyAllWindows()

    def isClosed(self):
        return self.close

    def setLines(self, line1, line2, line3):
        img = np.full((100,90,3),255, np.uint8)

        if(line1 > 0):
            line1 = 100 * (1 - line1)
            cv.rectangle(img,(15,100),(15, int(line1)),(0),20)

        if(line2 > 0):
            line2 = 100 * (1 - line2)
            cv.rectangle(img,(45,100),(45, int(line2)),(0),20)

        if(line3 > 0):
            line3 = 100 * (1 - line3)
            cv.rectangle(img,(75,100),(75, int(line3)),(0),20)

        self.overlay = img

    def videoLoop(self):
        """
        The mainloop
        """
        try:
            time.sleep(5)

            while(not self.isClosed()):
                frame = self.tello.read()
                #print(frame)
                #overlay
                overlay = self.overlay


                #height, width, depth= frame.shape
                #overlay = cv.resize(overlay,(width, height), cv.INTER_AREA)

                #print(frame.shape)
                #print(overlay.shape)
                #Overlay
                #cv.addWeighted(frame, 0.9, overlay, 0.1, 0.0, frame)
                cv.imshow('Lines', overlay)
                cv.imshow('TelloStream', frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            print("videoStream Exception")
            cv.destroyAllWindows()
