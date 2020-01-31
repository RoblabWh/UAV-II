import sys
import os
#print(os.getcwd())
sys.path.append(os.getcwd() + '/tello')
sys.path.append(os.getcwd() + '/user_interface')
#sys.path.append(os.getcwd() + '/network_interface')
sys.path.append('../network_interface')
sys.path.append(os.getcwd() + '/joystick')
#sys.path.append(sys.path + "/tello")
#print(sys.path)
import drone
import joystick
import ui
import network
import threading

def main():
    tello = drone.drone()
    joy = joystick.joystick(tello)
    vplayer = ui.UI(tello, joy)
    net_send = threading.Thread(target=network.video_server, args=(tello,))
    net_send.start()
    vplayer.root.mainloop()
    """a = []
    print(len(a))
    a.append(('192.168.178.154', 47765))
    a.append(('192.168.178.154', 47765))
    a.append(('192.168.178.154', 47765))
    a.append(('192.168.178.154', 47765))
    a.append(('192.168.178.154', 47765))
    print(a[0][0])"""


if __name__ == "__main__":
    main()