#import cv2 as cv
#import time
#import drone
#import threading


def main():
    """tello = drone.drone()
    #est = threading.Thread(target=show, args=(tello))
    #time.sleep(5)
    #test.start()
    time.sleep(3)
    tello.cmdTakeoff()
    time.sleep(30)
    print("lande jetzt")
    tello.cmdLand()
    time.sleep(5)
    tello.setTerminate()"""
    value = []
    description = []
    dic = {}
    test = "mid:-1;x:0;y:0;z:0;mpry:0,0,0;pitch:5;roll:13;yaw:-84;vgx:0;vgy:0;vgz:0;templ:73;temph:75;tof:10;h:0;bat:73;baro:68.86;time:0;agx:104.00;agy:-237.00;agz:-978.00;"
    print("Ausgabe: {}".format(test))
    test1 = test.split(";")
    for i in range(0,21):
        counts = test1[i].split(":")
        #print(counts[1])
        description.append((counts[0]))
        value.append(counts[1])
        dic[description[i]] = value[i]
    print(dic)


if __name__ == "__main__":
    main()
