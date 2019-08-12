import tello
import videoStream
from logger import write_to_log

import numpy as np
import cv2 as cv
import time
def main():
    write_to_log("Initialise Tello Drone")
    drone = tello.Tello('', 8889)
    vidStream = videoStream.videostream(drone)


    #Test
    time.sleep(20)
    vidStream.setLines(0.6,0,0.2)
    time.sleep(10)
    vidStream.setLines(0,0.9,0.1)


if __name__ == "__main__":
    main()
