# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 12:52:29 2020

@author: Marc
"""
import os
import numpy as np
import dataSetTransform_v00 as doDataSetTransform
import NYU_cameraParams
import cv2

transformedDatasetDirPath = r'E:\Neuer Ordner (2)'

if __name__=='__main__':
    
    cameraParams = NYU_cameraParams.getCameraParams()
    
    datasetDirNames = os.listdir(transformedDatasetDirPath)
    for datasetDirName in datasetDirNames:
        datasetDirPath = os.path.join(transformedDatasetDirPath, datasetDirName)
        subsetDirNames = os.listdir(datasetDirPath)
        for subsetDirName in subsetDirNames:
            subsetDirPath = os.path.join(datasetDirPath, subsetDirName)
            transDataDirNames = os.listdir(subsetDirPath)
            for transDataDirName in transDataDirNames:
                transDataDirPath = os.path.join(subsetDirPath, transDataDirName)
                transDataFileNames = os.listdir(transDataDirPath)
                
                pclCameraFound = False
                i = 0
                while not pclCameraFound:
                    dataFileName = transDataFileNames[i]
                    if dataFileName.endswith('pclCamera.npy'):
                        pclCameraFound = True
                    else:
                        i+=1
                
                if pclCameraFound:
                    pclCameraFilePath = os.path.join(transDataDirPath, transDataFileNames[i])
                    cameraPcl = np.load(pclCameraFilePath)
                    
                    projRgbdImg, finalPclIgnore = doDataSetTransform.projectTransfRgbdCameraImage(cameraPcl, cameraParams, (480,640))
                    
                    projRgbdImg *= (np.iinfo(np.uint16).max/10)
                    projRgbdImg = projRgbdImg[::2, ::2]
                    projRgbdImg = projRgbdImg.astype(np.uint16)
                    #projRgbdImg = projRgbdImg.byteswap()
                    
                    outpath = pclCameraFilePath[:pclCameraFilePath.index('pclCamera.npy')] + '_projectedNoOverflow.pgm'
                    print("Wrting D-Image...\n{}\n".format(outpath))
                    cv2.imwrite(outpath, projRgbdImg)
    