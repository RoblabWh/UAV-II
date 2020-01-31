# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 19:00:48 2019

@author: Marc
"""

import os
import numpy as np
import argparse

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import minmax_scale


def pathToNyp3DFile(canditate):
    
    pathToNypFile = None
    
    fileNotFound = True
    wrongFileFormat = True
    no3DFile = True
    
    if os.path.isfile(canditate):
        fileNotFound = False
        if canditate.endswith('.npy'):
            wrongFileFormat = False
            if len(np.load(canditate) == 3):
                no3DFile = False
                pathToNypFile = canditate
                
    if fileNotFound:
        raise argparse.ArgumentTypeError("{} file not found\n".format(canditate))
    elif wrongFileFormat:
        raise argparse.ArgumentTypeError("{} wrong data format\n".format(canditate))
    elif no3DFile:
        raise argparse.ArgumentTypeError(".nyp file does not contain an 3d depth image\n")
        
    return pathToNypFile
    
    
def main():
    
    
    #get args
    args = None
    path = focalLength = sensorWidth = sensorHeight = imageCenterHor = imageCenterVert = None
    
    parser = argparse.ArgumentParser(description='Get params')
    
    parser.add_argument('path', metavar='-p', type=pathToNyp3DFile)
    
    #parser = argparse.ArgumentParser(description='Enter camera Parameter <<Focal length>>')
    parser.add_argument('focallength', metavar='-f', type=float)
    
    #parser = argparse.ArgumentParser(description='Enter camera Parameter <<sensor width>>')
    parser.add_argument('sensorwidth', metavar='-w', type=float)
    
    #parser = argparse.ArgumentParser(description='Enter camera Parameter <<sensor height>>')
    parser.add_argument('sensorheight', metavar='-h', type=float)
    
    #parser = argparse.ArgumentParser(description='Enter camera Parameter <<image center hor.>>')
    parser.add_argument('imagecenterhor', metavar='-ch', type=int)
    
    #parser = argparse.ArgumentParser(description='Enter camera Parameter <<image center vert.>>')
    parser.add_argument('imagecentervert', metavar='-cv', type=int)
    
    args = parser.parse_args()
    path = args.path
    focalLength = args.focallength
    sensorWidth = args.sensorwidth
    sensorHeight = args.sensorheight
    imageCenterHor =  args.imagecenterhor
    imageCenterVert = args.imagecentervert
    
    #load file and check center params
    imgDepth = np.load(path)
    
    if (imageCenterHor < 0 or imageCenterHor > imgDepth.shape[1]) or (imageCenterVert < 0 or imageCenterVert > imgDepth.shape[0]):
            raise argparse.ArgumentTypeError("Center coordinates inconsisten with image size\n")
    
    
    #get 3D points
    points3D = None
    list3D = []
    print(imgDepth.shape)
    for i in range(0, imgDepth.shape[0]):
        for j in range(0, imgDepth.shape[1]):
            
            point3D = np.array([0.0,0.0,0.0])
            
            directionVector = np.array([0.0,0.0,0.0])   
            v = i - imageCenterVert
            u = j - imageCenterHor
            y = sensorHeight / imgDepth.shape[0] * v
            x = sensorWidth / imgDepth.shape[1] * u
            
            directionVector[0] = -x
            directionVector[1] = -y
            directionVector[2] = focalLength
            
            scaleFactor = imgDepth[i][j][0]
            scaleFactor = scaleFactor if scaleFactor < 5 else 0
            #if i == 0 and 100 < j < 300:
            #    print(imgDepth[i][j][0], "   ", directionVector * scaleFactor,"\n")
            point3D = directionVector
            point3D = point3D * scaleFactor * 20
            
            if point3D[2] < 0.2:
                list3D.append(point3D)
            
    points3D = np.array(list3D)
    
    fp = open('points.txt', 'w+')
    for j in range(imgDepth.shape[1]):
        fp.write('{}---{}\n'.format(j, imgDepth[6][j][0] if imgDepth[6][j][0] <5 else 0) )
            
    fp.close()
    #visualize points
    
    xData = yData = zData = None
    xData = [p[0] for p in list3D]
    yData = [p[1] for p in list3D]
    zData = [p[2] for p in list3D]
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #plt.scatter(xData, yData, zData)
    ax.plot(xData, yData, zData, 'r+')
    #ax = fig.add_subplot(112, projection='3d')
    sx = np.linspace(-sensorWidth, sensorWidth, 418)
    sy = np.linspace(-sensorHeight, sensorHeight, 118)
    po = []
    for x in sx:
        for y in sy:
            po.append([x,y,0])
    x2 = [p[0] for p in po]
    y2 = [p[1] for p in po]  
    z2 = [p[2] for p in po]    
    ax.plot(x2, y2, z2, 'b.')
    ax.scatter(0, 0, 0, 'b+')
    ax.scatter(0, 0, focalLength)
    ax.axis('equal')
    ax.set_xlabel('$X$', fontsize=20)
    ax.set_ylabel('$Y$', fontsize=20)
    ax.set_zlabel('$Z$', fontsize=20)
    
    plt.show()
    
    
if __name__=='__main__':
    main()