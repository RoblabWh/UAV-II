from PIL import Image, ImageDraw
from PIL import ImageTk
import Tkinter as tki
import threading
import time
import platform
import joystick
import cv2 as cv
import datetime
from multiprocessing import Process

class UI:
    """ Wrapper class to enable the GUI """

    def __init__(self, drone, joy):
        """ Init objectparameter """
        self.drone = drone
        self.frame = None
        self.video_thread = None
        self.stopEvent = None
        self.isFullscreen = False
        self.isRecord = False
        self.joy = joy
        #self.joy_thread = Process(target=self.joy.run)
        self.joy_thread = threading.Thread(target=self.joy.run)
        # initilaize record
        self.fps = 35
        self.fourcc = cv.VideoWriter_fourcc(*'XVID')
        self.videoStream = None

        # initialize the root window and image panel
        self.root = tki.Tk()
        self.panel = None

        # create buttons, checkboxes, etc.
        self.btn_fullscreen = tki.Button(self.root, text="Fullscreen einschalten", command=self.setFullscreen)
        self.btn_fullscreen.pack(side="bottom", fill="both", expand="no", padx=10, pady=5)

        self.btn_record = tki.Button(self.root, text="Record einschalten", command=self.toggleRecord)
        self.btn_record.pack(side="bottom", fill="both", expand="no", padx=10, pady=5)

        self.btn_land_takeoff = tki.Button(self.root, text="takeoff", command=self.landTakeoff)
        self.btn_land_takeoff.pack(side="bottom", fill="both", expand="no", padx=10, pady=5)

        self.btn_ai = tki.Button(self.root, text="AI einschalten", command=self.operatorControl)
        self.btn_ai.pack(side="bottom", fill="both", expand="no", padx=10, pady=5)

        """ 
            start a thread that constanly pools the video sensor for the most recently read frame
        """
        self.stopEvent = threading.Event()
        self.video_thread = threading.Thread(target=self.videoLoop, args=())
        self.video_thread.start()

        #set a callback to handle when the window is closed
        self.root.wm_title("Tello Controller")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

        # the sending_command will send command to tello every 5 seconds
        self.periodicSending_thread = threading.Thread(target=self._sendingCommand)

    def operatorControl(self):
        if(self.btn_ai['text'] == "AI einschalten"):
            #print("Joystick einschalten")
            self.joy_thread = self.joy_thread = threading.Thread(target=self.joy.run)
            self.joy_thread.start()
            #self.drone.set_joystick(True)
            self.btn_ai['text'] = "AI ausschalten"
            # AI aktivieren
        else:
            #print("Joystick ausgeschaltet")
            self.joy.turnoff()
            self.btn_ai['text'] = "AI einschalten"
            # AI deaktivieren

    def landTakeoff(self):
        if(self.btn_land_takeoff['text'] == "takeoff"):
            self.btn_land_takeoff['text'] = "land"
            self.drone.cmdTakeoff()
        else:
            self.btn_land_takeoff['text'] = "takeoff"
            self.drone.cmdLand()

    def setFullscreen(self):
        #cv.imshow("fullscreen", self.frame)
        #cv.setWindowProperty("fullscreen", cv.WND_PROP_AUTOSIZE, cv.CV_WINDOW_FULLSCREEN)
        self.btn_fullscreen['text'] = "Fullscreen ausschalten"
        self.isFullscreen = True


    def toggleRecord(self):
        if(self.isRecord):

            #Achtung das Release wirft einen grausamen fehler!!! villeicht in einen eigenen Process auslagern
            self.videoStream.release()
            self.btn_record['text'] = "Record einschalten"
            self.isRecord = False
        else:
            now = datetime.datetime.now()
            year = now.strftime('%Y')
            month = now.strftime('%m')
            day = now.strftime('%d')
            hour = now.strftime('%H')
            min = now.strftime('%M')
            sec = now.strftime('%S')
            name = '' + year + month + day + hour + min + sec
            self.videoStream = cv.VideoWriter('' + name + '.avi', self.fourcc, self.fps, (960, 720))  # (720, 960))
            self.btn_record['text'] = "Record ausschalten"
            self.isRecord = True
        #print self.frame.shape

    def _record(self, fps):
        if(self.isRecord):
            #print("schreibe stream")
            frame = self.frame
            b, g, r = cv.split(frame)
            frame = cv.merge((r,g,b))
            self.videoStream.write(frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                pass

    """Todo: Anpassen der text Ausrichtungen!!!"""
    def _getFrame(self, frame):
        text = []
        color = (0, 0, 0)
        #frame = self.frame
        width, height = frame.shape[:2]
        #print(height)
        width +=40
        height = 20
        #print(height)
        bat = self.drone.getBat()
        text.append("bat: {}".format(bat))
        h = self.drone.getH()
        text.append("h: {}".format(h))
        for i in range(0, len(text)):
            cv.putText(frame, text[i], (width, height), cv.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            height += 20
        return frame


    def videoLoop(self):
        """
                The mainloop thread of Tkinter
                Raises:
                    RuntimeError: To get around a RunTime error that Tkinter throws due to threading.
        """
        try:
            time.sleep(0.5)
            self.periodicSending_thread.start()
            #print("gestartet")
            while not self.stopEvent.is_set():
                system = platform.system()
                #print("after system")
                #read the frame for the GUI show
                self.frame = self.drone.getFrame()
                frame = self.frame.copy()
                if(frame is None or frame.size == 0):
                    continue

                frame = self._getFrame(frame)

                if(self.isFullscreen):
                    #print("drin")
                    cv.namedWindow("window", cv.WND_PROP_FULLSCREEN)
                    cv.setWindowProperty("window", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
                    #frame = self.frame
                    b, g, r = cv.split(frame)
                    frame = cv.merge((r, g, b))
                    cv.imshow("window", frame)
                    #print("a")
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        self.isFullscreen = False
                        self.btn_fullscreen['text'] = "Fullscreen einchalten"
                        cv.destroyAllWindows()
                        #break
                else:
                    #transfer the format from frame to Image
                    image = Image.fromarray(frame)
                    #draw = ImageDraw.Draw(image)
                    #draw.text((10,10), "hello World", fill=(255,0,0))
                    if system == "Windows" or system == "Linux":
                        #print("normal")
                        self._updateGUI(image)
                    else:
                        #print("start Thread in videoLoop")
                        thread_tmp = threading.Thread(target=self.updateGUI, args=(image,))
                        thread_tmp.start()
                        time.sleep(0.03)
                self._record(20)
        except RuntimeError:
            print("[INFO] caught a RuntimeError")

        #print("videoLoop ende")

    def _updateGUI(self, image):
        """
        Main operation to initial the object of image and update the GUI panel
        """
        image = ImageTk.PhotoImage(image)
        if (self.panel is None):
            self.panel = tki.Label(image=image)
            self.panel.image = image
            self.panel.pack(side="left", padx=10, pady=10)
        else:
            self.panel.configure(image=image)
            self.panel.image = image

    def _sendingCommand(self):
        """
        start a while loop that sends 'command' to tello every 5 second
        """

        while not self.stopEvent.is_set():
            self.drone.send_command('command')
            time.sleep(10)
            #print("gesendet")
        #print("sending ende")

    def onClose(self):
        """
        set the stop event, cleanup the camera, and allow the rest of

        the quit process to continue
        """
        print("[INFO] closing...")
        self.stopEvent.set()
        #self.drone.setTerminate()
        self.drone.__del__()
        del self.drone
        self.root.quit()
        print("[print end]")
