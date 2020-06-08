from tkinter import Tk
from tkinter import StringVar
from tkinter import OptionMenu
from tkinter import Label
from tkinter import Entry
from tkinter import Button
from tkinter import Listbox
from tkinter import END

import threading
import network
import re

import numpy as np
import cv2


# kommunikationsarten
OptionList = [
"send",
"get",
"both-UAV",
"both-Dist"
]

class ui:
    # Im Konstruktor wird die Gui aufgebaut
    def __init__(self):
        self.window = Tk()
        self.window.title("Distributer Network Konfigurator")
        self.rawData = None
        self.manipulatedData = None
        #Aufbau der GUI
        self._interface()


    def __del__(self):
        pass

    def mainloop(self):
        self.window.mainloop()

    # Nach einer bestimmten zeit, soll die Liste geupdatet werden, damit z.B.
    # der Ping angezeigt wird
    def updateList(self, name, ip, port, com, ping):
        liste = self.lstCom.get(1, END)
        found = False
        index = 1 # index von 1 wird benötigt, damit die erste Zeile stehen bleibt
        for lname in liste:
            lname = re.sub(' +', ' ', lname)
            lname = lname.split(' ')
            if(lname[1] == name):
                found = True
                self.lstCom.delete(index)
                txt = "{:^30} {:^30} {:^15} {:^20} {:^30}".format(name, ip, port, com, ping)
                self.lstCom.insert(index, txt)
                break
            else:
                index += 1
        return found

    """
    Wenn eine Verbindung aufgebaut werden soll, wird ein Thread erstellt und die
    eingetragenen Daten werden in einer Liste abgespeichert.
    """
    def _insert(self):
        #test eingabe
        #Bekomme die Daten
        ip = self.txt_ip.get()
        port = self.txt_port.get()
        name = self.txt_discribe.get()
        com = self.var.get()

        # Formatieren den String
        txt = "{:^30} {:^30} {:^15} {:^20} {:^30}".format(name, ip, port, com, "0ms")
        #print(txt)
        # Fuege ein
        self.lstCom.insert(END, txt)

        self.txt_discribe.delete(0, END)
        self.txt_port.delete(0, END)
        self.txt_ip.delete(0, END)

        # Starte Thread mit den Konfig einstellungen
        thread = threading.Thread(target=self._worker, args=(name, ip, port, com,))
        thread.start()

    """
    Hauptprogramm des erstellten Threads
    """
    def _worker(self, name, ip, port, com):
        net = network.network(name, ip, port, com)
        count = 0
        found = True
        rawData = None
        manipulatedData = None
        while(True):
            if((com == "send") or (com == "get")):
                counter, ping, rawData = net.run(self.getRawData())
            elif(com == "both-UAV"):
                #print(self.getManipulatedData() is None)
                counter, ping, rawData = net.run(self.getManipulatedData(), com)
                #print(rawData)
            elif(com == "both-Dist"):
                counter, ping, manipulatedData = net.run(self.getRawData(), com)
                #print(manipulatedData)

            if(counter >= 5):
                ping = "inf"
            else:
                ping = str(ping) + "ms"
            if(count == 350):
                count = 0
                found = self.updateList(name, ip, port, com, ping)

            if(not found):
                #cv2.destroyWindow(name)
                del(net)
                print("thread killed")
                break

            count += 1
            #self.rawData = rawData
            if(rawData is not None):
                self.setRawData(rawData)
            if(manipulatedData is not None):
                self.setManipulatedData(manipulatedData)


    def getRawData(self):
        return self.rawData

    def setRawData(self, data):
        self.rawData = data

    def getManipulatedData(self):
        return self.manipulatedData

    # Funktion zum Fusionieren der Rohdaten und der Manipulierten Daten
    # Es werden die beiden Bilder mit der Overlay-Funktion der opencv fusioniert
    def setManipulatedData(self, img):
        alpha = 0.3
        beta = 1.0 - alpha
        manipulatedData = img
        if(self.getRawData() is not None):
            # entschlüsseln der Daten
            array = np.frombuffer(self.getRawData(), np.dtype('uint8'))
            raw = cv2.imdecode(array, 1)
            # entschlüsseln der Daten
            array = np.frombuffer(manipulatedData, np.dtype('uint8'))
            man = cv2.imdecode(array, 1)
            # Overlay
            manipulatedData = cv2.addWeighted(raw, alpha, man, beta, 0.0)
            self.manipulatedData = manipulatedData
        if(manipulatedData is not None):
            array = np.frombuffer(img, np.dtype('uint8'))
            man = cv2.imdecode(array, 1)
            manipulatedData = cv2.addWeighted(man, alpha, man, beta, 0.0)
            self.manipulatedData = manipulatedData

    def _interface(self):
        '''
        Create a textbox for a name
        '''
        label_discribe = Label(text="Bezeichnung").grid(column=0, row=0)
        self.txt_discribe = Entry()
        self.txt_discribe.grid(column=0, row=1)

        '''
        Create a textbox for ip-adress
        '''
        label_ip = Label(text="IP").grid(column=1, row=0)
        self.txt_ip = Entry()
        self.txt_ip.grid(column=1, row=1)

        '''
        Create a textbox for port
        '''
        label_port = Label(text="Port").grid(column=2, row=0)
        self.txt_port = Entry()
        self.txt_port.grid(column=2, row=1)

        '''
        Create the option list
        '''
        label_com = Label(text="Kommunikationsart").grid(column=3, row=0)
        self.var = StringVar(self.window)
        self.var.set(OptionList[0]) # use "self.var.get()" to get the values of the list
        optMenu = OptionMenu(self.window, self.var, *OptionList)
        optMenu.config(width=8, font=('Helvetica', 12))
        optMenu.grid(column=3, row=1)

        '''
        Create button for listing
        '''
        btn_list = Button(text="eintragen", command=self._insert).grid(column=5, row=0, rowspan=2)

        '''
        Create a listbox for reprasentation the communication
        '''
        self.lstCom = Listbox(selectmode='browse', width=90)
        self.lstCom.grid(column=0, row=4, columnspan=6)
        txt_config = "{:^30} {:^30} {:^15} {:^20} {:^30}".format("Bezeichnung", "IP-Adresse", "Port", "Kommunikationsart", "Ping")
        self.lstCom.insert(1, txt_config)

        '''
        Create a button for close connections
        '''
        btn_close = Button(text="Verbindung trennen", command=self._closeConnection).grid(column=0, row=5, columnspan=6)

    # Schließt die ausgewaehlte Verbindung
    def _closeConnection(self):
        liste = self.lstCom.curselection()
        if(liste[0] != 0):
            self.lstCom.delete(liste[0])
