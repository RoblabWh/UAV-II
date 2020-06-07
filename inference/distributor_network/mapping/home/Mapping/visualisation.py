
import time
import os
import numpy as np


color_pipe = os.open("./color_fifo", os.O_RDONLY)
point_pipe = os.open("./point_fifo", os.O_RDONLY)
keyframe_pipe = os.open("./keyframe_fifo", os.O_RDONLY)

while True:
    keyframes = os.read(keyframe_pipe, 100000000)
    keyframes = np.frombuffer(keyframes, dtype=np.float32)
    keyframes = keyframes.reshape((-1, 3))
    points = os.read(point_pipe, 100000000)
    points = np.frombuffer(points, dtype=np.float32)
    points = points.reshape((-1, 3))
    color = os.read(color_pipe, 100000000)
    color = np.frombuffer(color, dtype=np.unit8)
    color = color.reshape((-1, 3))



    time.sleep(0.03)