# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 19:45:15 2019

@author: Marc Thurow, Gerhard Senkowski
"""

from imageio import imread
import cv2
import numpy as np
import os, shutil
import NYUCameraParams
from ListOfURLS import url_list
import time
import PlaneSegmentationAndLabeling as PlaneSegmentationAndLabeling

import requests
import zipfile, io

#baseInputDir= r'E:\Downloads2\basements\nyuPart1'
#baseInputDir= r'E:\Downloads2\living_rooms_part1'
baseInputDir = r'/media/gas/Samsung_T5/tmp3'
baseOutputDir = r'/media/gas/Samsung_T5/NYU_Dataset_Planes'
pathToFileCameraParams = r'E:\Downloads2\toolbox_nyu_depth_v2\camera_params.m'

def getSyncPathsToDatasetFiles(baseDir):
    '''
    This function walks through the baseDir and reads an Index.txt file. From here it gets the information
    which files are loaded eg. from the NYU-dataset ordered by the date of data acquisition. For every RGB-Image
    the corresponding Depth-Image is found by pairing the Images with the smallest time-based distance.
    '''
    syncPathsDatasetFiles = []
    
    subDirNames = os.listdir(baseInputDir)
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
                if syncRgbRgbd[0] is None:
                    if sensory.endswith('pgm'):
                        syncRgbRgbd[0] = os.path.join(subDir, sensory)
                elif syncRgbRgbd[1] is None:
                    if sensory.endswith('ppm'):
                        syncRgbRgbd[1] = os.path.join(subDir, sensory)
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
        Undistorts (rectifies) the input image.
        Note: The image types of rgb and depth input data differ and the output differs as well
    '''
    
    cameraMatrix = cameraParams[imgType]['pinhole']
    distCoeffs = cameraParams[imgType]['distCoeffs']
    
    '''undistort and rectify image '''
    newCameraMatrix = np.copy(cameraMatrix)
    map1,map2 = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, np.identity(3), newCameraMatrix,(img.shape[1],img.shape[0]),cv2.CV_32FC1)
    newImg = cv2.remap(img, map1, map2, interpolation=cv2.INTER_NEAREST)
    
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
        projP_u = int(projP[0]/projP[2])
        projP_v = int(projP[1]/projP[2])
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
    projRgbdImg = projRgbdImg[::2, ::2]
    finalPcl = finalPcl[::2,::2]
    finalPcl = finalPcl.reshape((finalPcl.shape[0]*finalPcl.shape[1],3))
    
    return projRgbdImg, finalPcl
    

def projectRgbdToRgb(rgbImg, rgbdImg, cameraParams):
    '''
    This function projects the unaligned Depth image passed by the parameter rgbdImg onto the Image plane of rgbImg and
    under consideration of the cameraParams-parameter and creates a new, aligned rgbdImg and PCL from it.
    '''
    projectedRgbd = np.zeros(shape=rgbdImg.shape)
    
    undistRgb = undistortImg(rgbImg, cameraParams, imgType='RGB')
    undistRgb = undistRgb.astype(np.uint8)
    undistRgbd = undistortImg(rgbdImg, cameraParams, imgType='RGBD') #depth values in unit meters, type float64
    
    pclKinectSpace = createPcl(undistRgbd, cameraParams)
    pclCameraSpace = poseTransformPclToCamera(pclKinectSpace, cameraParams)
    projectedRgbd, finalPcl = projectTransfRgbdCameraImage(pclCameraSpace, cameraParams, rgbdImg.shape)

    return undistRgb, projectedRgbd, pclKinectSpace, pclCameraSpace, projectedRgbd, finalPcl
    
def transformSingleSyncDataPackage(pathToSyncRgbRgbd, destPathSyncTrans):
    '''
    This function takes an 2-Tupel consistinc of paths to an RGB-image and the corresponding Depth-image from
    a Micrsoft Kinect and also a destination path, reads in the data, processes the data and saves the results at the
    destination path.
    '''
    outputDataSubDirPath = os.path.split(destPathSyncTrans[0])[0]
    if not os.path.isdir(outputDataSubDirPath):
        os.makedirs(outputDataSubDirPath)
    
    rgbImg = imread(pathToSyncRgbRgbd[1])
    rgbdImg = imread(pathToSyncRgbRgbd[0])
    rgbdImg = rgbdImg.astype(np.uint16)
    rgbdImg = rgbdImg.byteswap()
    cameraParams = CameraParams.getCameraParams()
    undistRgb, projectedRgbd, pclKinectSpace, pclCameraSpace, projectedRgbd, finalPcl = projectRgbdToRgb(rgbImg, rgbdImg, cameraParams)

    #save data
    cv2.imwrite(destPathSyncTrans[1], rgbImg)
    rgbdImg = cameraParams['DepthParams']['depthParam1'] / (cameraParams['DepthParams']['depthParam2'] - rgbdImg)

    rgbdImg *= 10000
    cv2.imwrite(destPathSyncTrans[0], rgbdImg.astype(np.uint16))
    
    undistRgbFilePath = destPathSyncTrans[1][:-4] + 'undist.ppm'
    cv2.imwrite(undistRgbFilePath, undistRgb)

    projectedRgbd *= 10000
    projectedRgbd = projectedRgbd.astype(np.uint16)
    projectedRgbdFilePath = destPathSyncTrans[0][:-4] + '_projected.pgm'
    cv2.imwrite(projectedRgbdFilePath, projectedRgbd)
    
    pclKinectFilePath = destPathSyncTrans[0][:-4] + 'pclKinect'
    np.save(pclKinectFilePath, pclKinectSpace)
    
    pclCameraFilePath = destPathSyncTrans[0][:-4] + 'pclCamera'
    np.save(pclCameraFilePath, pclCameraSpace)
    
    pclFinalPath = destPathSyncTrans[0][:-4] + 'pcl_final'
    np.save(pclFinalPath, finalPcl)

    PlaneLabeling.labelPlanes(pclFinalPath + '.npy', destPathSyncTrans[0][:-4])
    
    
def transformSyncDataFiles(baseOutputDir, syncDataPaths):
    '''
    Takes a path to an output dir and a list of path-tupels each of which stands either for a RGB image (.ppm-format)
    from a Microsoft Kinect  v1 or the correspondening Depth-image (.pgm-format).
    '''
    for pathToSyncRgbRgbd in syncDataPaths:
        destPathSyncTrans = getDestinationPathSyncTransformed(baseOutputDir, pathToSyncRgbRgbd)
        #print(destPathSyncTrans)
        transformSingleSyncDataPackage(pathToSyncRgbRgbd, destPathSyncTrans)

    
if __name__=='__main__':
    '''
    Loads NYU dataset and creates a new dataset from it
    Needs a connection to the internet and the URS in the imported URL lists must be up to date.
    For further information about the dataset creation please read the docs.
    
    '''
    input('Warning deletes content of baseInputdir.')
    zeroth_time = time.time()
    for url in url_list[26:]:
        first_time = time.time()
        r = requests.get(url)

        file = zipfile.ZipFile(io.BytesIO(r.content))
        file.extractall(baseInputDir)

        second_time = time.time()
        print('second - first', second_time - first_time)

        syncDataPaths = getSyncPathsToDatasetFiles(baseInputDir)
        syncDataPaths = syncDataPaths[::40]
        transformSyncDataFiles(baseOutputDir, syncDataPaths, pathToFileCameraParams)
        third_time = time.time()

        print('third - second', third_time - second_time)
        # from github https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder
        for filename in os.listdir(baseInputDir):
            file_path = os.path.join(baseInputDir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        # end
        fourth_time = time.time()
        print('fourth - third', fourth_time - third_time)
    fifth_time = time.time()
    print('fifth - zeroth', fifth_time -zeroth_time)

