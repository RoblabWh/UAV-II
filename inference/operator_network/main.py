import sys
import os
#print(os.getcwd())
# Fuegt die einzelnen Pfade dem Programm hinzu, damit die Module gefunden werden können
sys.path.append(os.getcwd() + '/tello')
sys.path.append(os.getcwd() + '/user_interface')
#sys.path.append(os.getcwd() + '/network_interface')
sys.path.append('../network_interface')
sys.path.append(os.getcwd() + '/joystick')
sys.path.append(os.getcwd() + '/navigation')
#sys.path.append(sys.path + "/tello")
#print(sys.path)
import drone
import joystick
import ui
import network
import threading
import ai

def main():
    tello = drone.drone()
    nav = ai.ai(tello)
    joy = joystick.joystick(tello)
    vplayer = ui.UI(tello, joy, nav)
    net_send = threading.Thread(target=network.video_server, args=(tello,))
    net_send.start()
    vplayer.root.mainloop()

if __name__ == "__main__":
    main()
