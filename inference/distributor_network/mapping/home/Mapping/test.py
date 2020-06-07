import numpy as np
import open3d as o3d
from numba import jit
import math
import cv2
import video_inference

out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (960,720))
cap = cv2.VideoCapture("/media/gas/1tbssd/Videos/tello_szenario_new.avi")

img_count = 0
segmentation = video_inference.Segmentation()
# with open("./test.txt", "wb") as image_fifo:
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    alpha = 0.5
    beta = (1.0 - alpha)
    # Display the resulting frame
    dst = cv2.addWeighted(frame, alpha, segmentation.get_segment(frame), beta, 0.0)
    cv2.imshow('frame', dst)
    out.write(dst)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
