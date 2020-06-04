# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 19:45:15 2019

@author: Marc Thurow, Gerhard Senkowski
"""

from imageio import imread
import cv2
import numpy as np
import os, shutil
import CameraParams
from ListOfURLS import url_list
import time
import PlaneSegmentationAndLabeling as PlaneSegmentationAndLabeling

import requests
import zipfile, io

import multiprocessing as mp

#baseInputDir= r'E:\Downloads2\basements\nyuPart1'
#baseInputDir= r'E:\Downloads2\living_rooms_part1'
baseInputDir= r'E:\Downloads2\kinectDataP'
#baseInputDir = r'/media/gas/Samsung_T5/tmp/'
#baseOutputDir = r'E:\transformedNYUDataset'
baseOutputDir = r'E:\Downloads2\kinectTransP'

def getSyncPathsToDatasetFiles(baseDir):
    '''
    This function walks through the baseDir and reads an Index.txt file. From here it gets the information
    which files are loaded eg. from the NYU-dataset ordered by the date of data acquisition. For every RGB-Image
    the corresponding Depth-Image is found by pairing the Images with the smallest time-based distance.
    '''
    syncPathsDatasetFiles = []
    
    subDirNames = os.listdir(baseDir)
    for subDirName in subDirNames:
        subDir = os.path.join(baseDir, subDirName)
        files = os.listdir(subDir)
        if 'INDEX.txt' in files:
            indexFileIndex = files.index('INDEX.txt')
        elif 'index.txt' in files:
            indexFileIndex = files.index('index.txt')
        else:
            continue
        indexFileName = files[indexFileIndex]
        with open(os.path.join(subDir, indexFileName), 'r') as iF:
            sensoryDataOrderedByTime = iF.read().split()
            syncRgbRgbd = [None,None]
            for sensory in sensoryDataOrderedByTime:
                if sensory.endswith('pgm'):
                    if syncRgbRgbd[0] is None:
                        syncRgbRgbd[0] = os.path.join(subDir, sensory)
                elif sensory.endswith('ppm'):
                    if syncRgbRgbd[1] is None:
                        syncRgbRgbd[1] = os.path.join(subDir, sensory)
                
                if syncRgbRgbd[0] is not None and syncRgbRgbd[1] is not None:
                    syncPathsDatasetFiles.append(syncRgbRgbd)
                    syncRgbRgbd = [None,None]
                    
    return syncPathsDatasetFiles

def getDestinationPathSyncTransformed(baseOutputDir, pathToSyncRgbRgbd):
    '''
    This funtction taktes the base output dir and a 2-Tupel indicating the RGB-image and the corresponding depth-image
    of a Microsoft Kinect v1 and produces a new path string for the output after data processing.
    '''
    import re
    
    destPathSyncTrans = []
    #print(pathToSyncRgbRgbd)
    inputDataSubDirPath = os.path.split(pathToSyncRgbRgbd[0])[0]
    rgbFileName = os.path.split(pathToSyncRgbRgbd[0])[-1]
    rgbdFileName = os.path.split(pathToSyncRgbRgbd[1])[-1]
    
    rgbTimestamp = rgbFileName[2:-4]
    rgbTimestamp = re.split('[.-]', rgbTimestamp)
    rgbTimestamp = rgbTimestamp[0] + '_' + rgbTimestamp[1]
    
    rgbdTimestamp = rgbdFileName[2:-4]
    rgbdTimestamp = re.split('[.-]', rgbdTimestamp)
    rgbdTimestamp = rgbdTimestamp[0] + '_' + rgbdTimestamp[1]
    outputDataSubDirName = 'rgb{}_rgbd{}'.format(rgbTimestamp, rgbdTimestamp)
    outputDataSubDirPath = os.path.join(baseOutputDir, os.path.split(inputDataSubDirPath)[-1])
    outputDataSubDirPath = os.path.join(outputDataSubDirPath, outputDataSubDirName)
    destPathSyncTrans.append(os.path.join(outputDataSubDirPath, rgbFileName))
    destPathSyncTrans.append(os.path.join(outputDataSubDirPath, rgbdFileName))
    
    return destPathSyncTrans

def applyDistortion(rays, cameraParams):
    
    distortedRays = None
    
    return distortedRays
    
def undistortImg(img, cameraParams, imgType='RGB'):
    '''
        Undistort (rectify) the input image.
        Note: The image types of rgb and depth input data differ and the output as well
    '''
    
    cameraMatrix = cameraParams[imgType]['pinhole']
    distCoeffs = cameraParams[imgType]['distCoeffs']
    
    '''undistort and rectify image '''
    newCameraMatrix = np.copy(cameraMatrix)
    map1,map2 = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, np.identity(3), newCameraMatrix,(img.shape[1],img.shape[0]),cv2.CV_32FC1)
    newImg = cv2.remap(img, map1, map2, interpolation=cv2.INTER_NEAREST)
    newImg = np.copy(img)
    
    if imgType=='RGBD':
        noiseMask = np.zeros(shape=(img.shape),dtype=np.float64)
        noiseMask[np.where(img==np.max(img))] = 255
        dstNoiseMap=cv2.remap(noiseMask, map1, map2, interpolation=cv2.INTER_NEAREST)
        newImg[np.where(dstNoiseMap==255)] = 2047 #noise
        newImg[np.where(newImg<600)] = 2047 #also noise
        
        #data currently in kinect specific local coordinate system
        #transfer to unit meters
        maxDepth = cameraParams['DepthParams']['maxDepth']
        depthParam1 = cameraParams['DepthParams']['depthParam1']
        depthParam2 = cameraParams['DepthParams']['depthParam2']
        
        imgDepthAbs = depthParam1 / (depthParam2 - newImg)
        imgDepthAbs[np.where(imgDepthAbs>maxDepth)] = maxDepth
        imgDepthAbs[np.where(imgDepthAbs<0)] = 0
        
        newImg = imgDepthAbs
        
    return newImg

def createPcl(undistRgbd, cameraParams):
    '''
    This function takes an Depth image of the Microsoft Kinect v1 and creates and returns a PCL using the
    camera parameters of the Kinect.
    '''
    #pcl = np.zeros(shape=(undistRgbd.shape[0], undistRgbd.shape[1], 3)) #480x640x3
    pcl = None
    cameraMatrix = cameraParams['RGBD']['pinhole']
    
    x = np.arange(0,undistRgbd.shape[1])
    y = np.arange(0,undistRgbd.shape[0])
    xx,yy = np.meshgrid(x, y)
    
    cx = cameraMatrix[0,2]
    cy = cameraMatrix[1,2]
    fx = cameraMatrix[0,0]
    fy = cameraMatrix[1,1]
    X = (xx - cx) * (undistRgbd / fx)
    Y = (yy - cy) * (undistRgbd / fy)
    Z = undistRgbd
    
    pcl = np.stack([X,Y,Z],axis=2)
    pcl=np.reshape(pcl, (pcl.shape[0]*pcl.shape[1],3))
    
    return pcl
    
def poseTransformPclToCamera(pclKinectSpace, cameraParams):
    '''
    Performs an SO3 transform to transform the PCL calculated from the depth image of a Kinect to the camera coordinate
    system of the RGB-Image sensor.
    '''
    transformedPcl = None
    
    R = cameraParams['RGBD_To_RGB']['R']
    t = cameraParams['RGBD_To_RGB']['t']
    
    transformedPcl = [np.dot(R, p) - np.dot(R, t) for p in pclKinectSpace]
    
    return transformedPcl

def projectTransfRgbdCameraImage(pclCameraSpace, cameraParams, shapeImg):
    '''
    This function projects a PCL onto an image plane given the camera Parameters
    '''
    projRgbdImg = np.zeros(shapeImg)
    cameraMatrix = cameraParams['RGB']['pinhole']
    finalPcl = np.zeros(shape=(shapeImg[0],shapeImg[1],3))

    for p in pclCameraSpace:
        projP = np.dot(cameraMatrix, p)
        projP_u = int(round(projP[0]/projP[2]))
        projP_v = int(round(projP[1]/projP[2]))
        projP_d = projP[2] #in meter
        if 0 < projP_u < shapeImg[1] and 0 < projP_v < shapeImg[0]:
            currentDepth = projRgbdImg[projP_v, projP_u]
            if currentDepth > 0:
                if currentDepth > projP_d:
                    projRgbdImg[projP_v, projP_u] = projP_d
                    finalPcl[projP_v, projP_u] = p

            elif currentDepth==0:
                projRgbdImg[projP_v, projP_u] = projP_d
                finalPcl[projP_v, projP_u] = p


    projRgbdImg[np.where(projRgbdImg>cameraParams['DepthParams']['maxDepth'])]=cameraParams['DepthParams']['maxDepth']
    #nan check?
    projRgbdImg[np.where(np.isnan(projRgbdImg))]=0
    
    return projRgbdImg, finalPcl
    

def projectRgbdToRgb(rgbImg, rgbdImg, cameraParams):
    '''
    This function projects the unaligned Depth image passed by the parameter rgbdImg onto the Image plane of rgbImg and
    under consideration of the cameraParams-parameter and creates a new, aligned rgbdImg and PCL from it.
    '''
    projectedRgbd = np.zeros(shape=rgbdImg.shape)
    
    #undistRgb = np.copy(rgbImg)
    undistRgb = undistortImg(rgbImg, cameraParams, imgType='RGB')
    undistRgb = undistRgb.astype(np.uint8)
    undistRgbd = undistortImg(rgbdImg, cameraParams, imgType='RGBD') #depth values in unit meters, type float64
    #undistRgbd = np.copy(rgbdImg)
    
    pclKinectSpace = createPcl(undistRgbd, cameraParams)
    pclCameraSpace = poseTransformPclToCamera(pclKinectSpace, cameraParams)
    projectedRgbd, finalPcl = projectTransfRgbdCameraImage(pclCameraSpace, cameraParams, rgbdImg.shape)

    return undistRgb, projectedRgbd, pclKinectSpace, pclCameraSpace, projectedRgbd, finalPcl
    
def transformSingleSyncDataPackage(pathToSyncRgbRgbd, destPathSyncTrans, flog):
    '''
    This function takes an 2-Tupel consistinc of paths to an RGB-image and the corresponding Depth-image from
    a Micrsoft Kinect and also a destination path, reads in the data, processes the data and saves the results at the
    destination path.
    '''
    outputDataSubDirPath = os.path.split(destPathSyncTrans[0])[0]
    if not os.path.isdir(outputDataSubDirPath):
        os.makedirs(outputDataSubDirPath)
    
    cameraParams = NYU_cameraParams.getCameraParams()
    
    rgbImg = imread(pathToSyncRgbRgbd[1])
    rgbdImg = imread(pathToSyncRgbRgbd[0])
    rgbdImg = rgbdImg.astype(np.uint16)
    rgbdImg[np.where(rgbdImg > 10000)] = 10000
    rgbdImg[np.where(rgbdImg == 0)] = 10000
    rgbdImgRel = rgbdImg / 1000
    rgbdImgRel = cameraParams['DepthParams']['depthParam2'] - cameraParams['DepthParams']['depthParam1'] / rgbdImgRel  # to relative
    
    #rgbdImg = rgbdImg.byteswap()
    undistRgb, projectedRgbd, pclKinectSpace, pclCameraSpace, projectedRgbd, finalPcl = projectRgbdToRgb(rgbImg, rgbdImgRel, cameraParams)

    #save data
    cv2.imwrite(destPathSyncTrans[1], rgbImg)
    #rgbdImg = cameraParams['DepthParams']['depthParam1'] / (cameraParams['DepthParams']['depthParam2'] - rgbdImg)

    #rgbdImg *= (np.iinfo(np.uint16).max/10000)
    
    
    rgbdImg *= 6
    cv2.imwrite(destPathSyncTrans[0], rgbdImg.astype(np.uint16))
    #cv2.imwrite(destPathSyncTrans[0], rgbdImg
    
    undistRgbFilePath = destPathSyncTrans[1][:-4] + 'undist.ppm'
    cv2.imwrite(undistRgbFilePath, undistRgb)
    
    projectedRgbd = projectedRgbd[::2, ::2]
    projectedRgbd *= (np.iinfo(np.uint16).max/10)
    projectedRgbd = projectedRgbd.astype(np.uint16)
    projectedRgbdFilePath = destPathSyncTrans[0][:-4] + '_projected.pgm'
    cv2.imwrite(projectedRgbdFilePath, projectedRgbd)
    
    pclKinectFilePath = destPathSyncTrans[0][:-4] + 'pclKinect'
    np.save(pclKinectFilePath, pclKinectSpace)
    
    pclCameraFilePath = destPathSyncTrans[0][:-4] + 'pclCamera'
    np.save(pclCameraFilePath, pclCameraSpace)
    
    finalPcl = finalPcl[::2,::2]
    finalPcl = finalPcl.reshape((finalPcl.shape[0]*finalPcl.shape[1],3))
    pclFinalPath = destPathSyncTrans[0][:-4] + 'pcl_final'
    np.save(pclFinalPath, finalPcl)

    planesLabeled = PlaneLabeling.labelPlanes(pclFinalPath + '.npy', destPathSyncTrans[0][:-4])
    if not planesLabeled:
        flog.write("[{}]:\t{}\nNo planes Found!".format(time.strftime('%y-%m-%d %H:%M:%S'), pathToSyncRgbRgbd[0]))
    
    
def transformSyncDataFiles(baseOutputDir, syncDataPaths,q):
    '''
    Takes a path to an output dir and a list of path-tupels each of which stands either for a RGB image (.ppm-format)
    from a Microsoft Kinect  v1 or the correspondening Depth-image (.pgm-format).
    '''
    destPathSyncTranses = []
    with open(os.path.join(baseOutputDir, 'log.txt'), 'w') as flog:
        
        for pathToSyncRgbRgbd in syncDataPaths:
            #print(pathToSyncRgbRgbd)
            destPathSyncTrans = getDestinationPathSyncTransformed(baseOutputDir, pathToSyncRgbRgbd)
            destPathSyncTranses.append(destPathSyncTrans)
            
            #print(destPathSyncTrans)
            transformSingleSyncDataPackage(pathToSyncRgbRgbd, destPathSyncTrans, flog)
    q.put(destPathSyncTrans)
    
if __name__=='__main__':
    '''
    Loads dataset and creates a new dataset from it
    For further information about the dataset creation please read the docs.
    '''
    #input('Warning deletes content of baseInputdir.')
    #zeroth_time = time.time()
    
    mp.set_start_method('spawn')
    
    second_time = time.time()
    
    baseInputDirP1 = baseInputDir + '1'
    baseInputDirP2 = baseInputDir + '2'
    baseInputDirP3 = baseInputDir + '3'
    baseInputDirP4 = baseInputDir + '4'
    
    syncDataPathsP1 = getSyncPathsToDatasetFiles(baseInputDirP1)
    syncDataPathsP2 = getSyncPathsToDatasetFiles(baseInputDirP2)
    syncDataPathsP3 = getSyncPathsToDatasetFiles(baseInputDirP3)
    syncDataPathsP4 = getSyncPathsToDatasetFiles(baseInputDirP4)
    
    baseOutputDirP1 = baseOutputDir + '1'
    baseOutputDirP2 = baseOutputDir + '2'
    baseOutputDirP3 = baseOutputDir + '3'
    baseOutputDirP4 = baseOutputDir + '4'
    q1 = mp.Queue()
    q2 = mp.Queue()
    q3 = mp.Queue()
    q4 = mp.Queue()
    p1 = mp.Process(target=transformSyncDataFiles,args=(baseOutputDirP1,syncDataPathsP1,q1,))
    p2 = mp.Process(target=transformSyncDataFiles,args=(baseOutputDirP2,syncDataPathsP2,q2,))
    p3 = mp.Process(target=transformSyncDataFiles,args=(baseOutputDirP3,syncDataPathsP3,q3,))
    p4 = mp.Process(target=transformSyncDataFiles,args=(baseOutputDirP4,syncDataPathsP4,q4,))
    #transformSyncDataFiles(baseOutputDir, syncDataPaths)
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    
    
    
    s1 = q1.get()
    s2 = q2.get()
    s3 = q3.get()
    s4 = q4.get()
    
    print(s1)
    print()
    print()
    print(s2)
    print()
    print()
    print(s3)
    print()
    print()
    print(s4)
    
    p1.join()
    p2.join()
    p3.join()
    p4.join()

