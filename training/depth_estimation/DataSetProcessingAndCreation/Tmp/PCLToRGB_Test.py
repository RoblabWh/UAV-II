# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 15:53:42 2020

@author: Marc
"""

import numpy as np
import NYU_cameraParams
import matplotlib.pyplot as plt

def projectPclToRGB_TEST(pcl, cameraParams, imageShape):
    projRgbdImg = np.zeros(imageShape)
    cameraMatrix = cameraParams['RGB']['pinhole']
    finalPcl = np.zeros(shape=(imageShape[0],imageShape[1],3))
    sliceForLoggingReal = []
    sliceForLoggingInt = []
    sliceForLoggingP = []
    for p in pcl:
        projP = np.dot(cameraMatrix, p)
        projP_u = int(round(projP[0]/projP[2]))
        projP_v = int(round(projP[1]/projP[2]))
        
        if imageShape[0]//2 == projP_v:
            sliceForLoggingReal.append(projP[0]/projP[2])
            sliceForLoggingInt.append(int(round(projP[0]/projP[2])))
        
        projP_d = projP[2] #in meter
        if 0 < projP_u < imageShape[1] and 0 < projP_v < imageShape[0]:
            currentDepth = projRgbdImg[projP_v, projP_u]
            if currentDepth > 0:
                if currentDepth > projP_d:
                    projRgbdImg[projP_v, projP_u] = projP_d
                    finalPcl[projP_v, projP_u] = p

            elif currentDepth==0:
                projRgbdImg[projP_v, projP_u] = projP_d
                finalPcl[projP_v, projP_u] = p
 
    projRgbdImg[np.where(projRgbdImg>cameraParams['DepthParams']['maxDepth'])]=cameraParams['DepthParams']['maxDepth']
    projRgbdImg[np.where(np.isnan(projRgbdImg))]=0
    return sliceForLoggingReal, sliceForLoggingInt, sliceForLoggingP

pathCameraPcl = r'E:\transformedNYUDataset\basement_0001a\rgb1316653580_471513_rgbd1316653580_484909\d-1316653580.471513-1316138413pclCamera.npy'
cameraPcl = np.load(pathCameraPcl)

cameraParams = NYU_cameraParams.getCameraParams()

sliceForLoggingReal, sliceForLoggingInt, sliceForLoggingP = projectPclToRGB_TEST(cameraPcl, cameraParams, (480,640))

''' RED : first derivative of real values, BLUE: first derivative of the discretized values '''
plt.figure(111)
x = np.arange(0,len(sliceForLoggingReal)-1)
a = sliceForLoggingReal[1:]
b = sliceForLoggingReal[:-1]
y = [a[i]-b[i] for i in range(len(a))]
plt.plot(x,y,c='red')

x2 = np.arange(0,len(sliceForLoggingInt)-1)
a2 = sliceForLoggingInt[1:]
b2 = sliceForLoggingInt[:-1]
y2 = [a2[i]-b2[i] for i in range(len(a))]
plt.plot(x2,y2,c='blue')
plt.show()
