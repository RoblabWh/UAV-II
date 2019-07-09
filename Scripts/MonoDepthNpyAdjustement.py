# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:11:21 2019

@author: Marc
"""

import numpy as np
import os
import sys
from tempfile import TemporaryFile
from shutil import copyfile
import argparse
import imageio

def monoDepthNpy(candidate):
    return candidate
    
def outPathValid(pathCanditate):
    try:
        testfile = os.path.join(pathCanditate, 'test.txt')
        f = open(testfile, 'w')
        f.close()
        os.remove(testfile)
    except Exception:
        raise Exception("Error at Creation/Deletion of Test File for the Path you entered\n")
        
    return pathCanditate
        
def main():
    
    inPath = None
    outPath = None
    
    parser = argparse.ArgumentParser(description='Get params')
    parser.add_argument('inputpath', metavar='-i', type=monoDepthNpy)
    parser.add_argument('outputpath', metavar='-o', type=outPathValid)
    
    arguments = parser.parse_args()
    inPath = arguments.inputpath
    outPath = arguments.outputpath
    print(inPath)
    print(outPath)
    print()
    print()
        
    #scan for npy files
    filelist = os.listdir(inPath)
    npylist = [file for file in filelist if file.endswith('.npy')]
    
    #format npy and save
    for npyFile in npylist:
        
        npArray = np.load(os.path.join(inPath, npyFile))
        npArray = np.squeeze(npArray)
        width, heigth = npArray.shape
        newNpArray = np.zeros((heigth, width, 1))
        newNpArray = np.array([[col for col in row] for row in npArray])
        
        np.save(os.path.join(outPath, os.path.split(npyFile)[1]), newNpArray)
        
        fileRgbName_disp = npyFile.split('.')[0]
        imgInputName = fileRgbName_disp+ '.jpg'
        fileRgbName_dispIndex = fileRgbName_disp.index('_disp')
        imgOutputName = imgInputName[:fileRgbName_dispIndex] + '.png'

        imgInputPath = os.path.join(inPath, imgInputName)
        imgOutputPath = os.path.join(outPath, imgOutputName)

        imgInput = imageio.imread(imgInputPath)
        imageio.imwrite(imgOutputPath, imgInput)
        
    print("well done ... goodbye")
        
    
if __name__=='__main__':
    main()