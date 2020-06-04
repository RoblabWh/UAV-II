# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 09:49:04 2019
​
@author: Marc Thurow, Gerhard Senkowski

"""
import math
import os

import numpy as np
import time
from numba import jit
import scipy.ndimage.morphology as morph
import random
import copy
import cv2
from imageio import imread


expectedRelPointDistanceUnderNoise = 0.1


@jit(nopython=True)
def cross(a, b):
    c = np.ones(3)
    c[0] = a[1] * b[2] - a[2] * b[1]
    c[1] = a[2] * b[0] - a[0] * b[2]
    c[2] = a[0] * b[1] - a[1] * b[0]
    return c

@jit(nopython=True)
def norm(vec):
    return math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)


@jit(nopython=True)
def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

@jit(nopython=True)
def absolute(a):
    return a if a > 0 else -a


def getSimPCLFromObject(resx, resy, objectType='plane', step=False):
    
    #320,240
    x = np.arange(0, resx, dtype=int)
    y = np.arange(0, resy, dtype=int)
    
    xx, yy = np.meshgrid(x, y)
    
    zz = None
    if objectType == 'plane':
        zz = np.ones((xx.shape))
    elif objectType == 'noisyPlane':
        zz = np.ones((xx.shape))
        gss = np.random.normal(size=zz.shape)
        zz = zz * gss * 0.1
    elif objectType == 'hemisphere':
        alpha = 0.25*np.min((resx,resy))
        alphaSquared = alpha**2
        zz = np.ones(xx.shape)
        zz = np.sqrt(np.abs(alphaSquared - (xx - 0.5 * resx)**2 - (yy - 0.5 * resy)**2))
        zz[np.where(np.sqrt((xx - 0.5 * resx)**2 + (yy - 0.5 * resy)**2) > alpha)] = 0
        zz = zz + 100
    elif objectType=='NoisyHemisphere':
        alpha = 0.25*np.min((resx,resy))
        alphaSquared = alpha**2
        zz = np.ones(xx.shape)
        zz = np.sqrt(np.abs(alphaSquared-(xx-0.5*resx)**2-(yy-0.5*resy)**2))
        zz[np.where(np.sqrt((xx-0.5*resx)**2+(yy-0.5*resy)**2) > alpha)] = 0
        zz = zz + 100
        gss = np.random.normal(size=zz.shape)
        zz = (zz + gss*0.4)
    else:
        raise Exception("Object Type {} not yet supported! Goodbye".format(objectType))
        
    zz2 = np.zeros(shape=(xx.shape[0], xx.shape[1], 3))
    for r in range(xx.shape[0]):
        for c in range(xx.shape[1]):  
            zz2[r, c] = np.array((xx[r, c], yy[r, c], zz[r, c]))
            
    return xx,yy,zz2


def binaCurvImg(curvImg, threshold=0.03):
    
    measureCurvBin = np.zeros(shape=curvImg.shape,dtype=int)
    for y in range(curvImg.shape[0]):
        for x in range(curvImg.shape[1]):
            if np.abs(curvImg[y,x]) < threshold:
                measureCurvBin[y,x] = 1
                    
    return measureCurvBin


def segmentBinCurvImg(binCurvImg, minSegmentSizeRel=0.025):
    
    segments = []
    
    thesisSet = set()
    for r, row in enumerate(binCurvImg):
        for p, binPixel in enumerate(row):
            if binPixel == 1:
                thesisSet.add((r, p))
    
    while len(thesisSet) > 0:
        primeSeed = thesisSet.pop()
        regionSet = set()
        workingSet = set()
        regionSet.add(primeSeed)
        workingSet.add(primeSeed)
        while len(workingSet) > 0:
            currentSeed = workingSet.pop()
            v, u = currentSeed
            if binCurvImg.shape[1]-1 > u >= 0:
                if binCurvImg[v, u+1] == 1:
                    right = (currentSeed[0], currentSeed[1]+1)
                    if right not in regionSet:
                        workingSet.add(right)
                        regionSet.add(right)
            if binCurvImg.shape[1] >= u > 1:
                if binCurvImg[v, u-1] == 1:
                    left = (currentSeed[0], currentSeed[1]-1)
                    if left not in regionSet:
                        regionSet.add(left)
                        workingSet.add(left)
            if binCurvImg.shape[0] >= v > 1:
                if binCurvImg[v-1, u] == 1:
                    up = (currentSeed[0]-1, currentSeed[1])
                    if up not in regionSet:
                        regionSet.add(up)
                        workingSet.add(up)
            if binCurvImg.shape[0]-1 > v >= 0:
                if binCurvImg[v+1, u] == 1:
                    down = (currentSeed[0]+1, currentSeed[1])
                    if down not in regionSet:
                        regionSet.add(down)
                        workingSet.add(down)
        
        thesisSet = thesisSet - regionSet
        imgSize = binCurvImg.shape[0]*binCurvImg.shape[1]
        if len(regionSet) >= minSegmentSizeRel * imgSize:
            segments.append(regionSet)
    
    return segments


def getThreeRandomPoints(points):

    randomIndices = random.sample(range(0, len(points)), 3)
    return points[randomIndices[0]], points[randomIndices[1]], points[randomIndices[2]]


def ransacPlane(segment, localNormals):
    
    filteredSegment = []
    
    superSegment3D = []
    superSegment2D3D = []
    for s in segment:
        superSegment2D3D.append((s, localNormals[s[0],s[1],8,0]))
        superSegment3D.append(localNormals[s[0],s[1],8,0])
        
    outlier_q = 0.4
    p = 0.999
    epsilon = 0.02
    k = np.log(1-p)/np.log(1-(1-outlier_q)**3)
    k = int(np.ceil(k))
    
    bestConsensus = []
    bestNormal = None
    consensusSet = None
    for i in range(k):
        p1, p2, p3 = getThreeRandomPoints(superSegment3D)
        normal = np.cross(np.array(p2)-np.array(p1), np.array(p3) - np.array(p1))
        normal = normal / norm(normal)
        dPlane = np.dot(normal, p1)
        consensusSet = set()
        for pTupel in superSegment2D3D:
            p = pTupel[1]
            d = np.abs(dot(normal, np.array(p)) - dPlane)
            if d < epsilon:
                pHashable = ((pTupel[0][0], pTupel[0][1]), (pTupel[1][0], pTupel[1][1], pTupel[1][2]))
                consensusSet.add(pHashable)
                

        if len(consensusSet) > len(bestConsensus):
            bestConsensus = copy.deepcopy(consensusSet)
            bestNormal = normal
            bestPlaneDistance = dPlane

    filteredSegment = list(bestConsensus)
    
    return filteredSegment, (bestNormal, bestPlaneDistance)


def filterOutlierSegments(segments, localNormals):

    filteredSegments = []

    planes = []

    for segment in segments:
        filteredSegment, plane = ransacPlane(segment, localNormals)
        filteredSegmentImg = np.zeros(shape=(localNormals.shape[0], localNormals.shape[1]))
        for fs in filteredSegment:
            pCoo = fs[0]
            filteredSegmentImg[pCoo[0], pCoo[1]] = 1

        filteredSegmentSub = segmentBinCurvImg(filteredSegmentImg, minSegmentSizeRel=0.025)
        if len(filteredSegmentSub) > 0:
            filteredSegments.append(filteredSegmentSub[0])
            planes.append(plane)

    return filteredSegments, planes


def plane_intersection(LP0, LP1, n, d):
    # http://geomalgorithms.com/a05-_intersect-1.html
    u = LP1 - LP0
    w = LP0 - (n * d)
    D = dot(n, u)
    N = -dot(n, w)
    if abs(D) < 0.01:
        return [0]
    sI = N / D
    #if sI < 0 or sI > 1:
    #    return 'not in front'

    I = LP0 + sI * u
    return I

def checkDistanceToPlane(p, plane,threshold=0.02):
    return absolute(dot(p, plane[0])-plane[1]) > threshold


def labelPlanes(pointcloudPath, outputPathWithFileBasename):
    
    planesDiscoveredAndLabeled = False
    
    t1 = time.time()

    pcl = np.load(pointcloudPath)

    x = np.arange(0,320,dtype=int)
    y = np.arange(0,240,dtype=int)
    xx, yy = np.meshgrid(x,y)
    zzOrig = np.zeros(shape=(xx.shape[0], xx.shape[1], 3))
    for i, row in enumerate(zzOrig):
        for j,p in enumerate(row):
            index = i*zzOrig.shape[1]+j
            zzOrig[i, j] = pcl[index]

    xx = xx[::4,::4]
    yy = yy[::4,::4]
    zz = zzOrig[::4,::4]

    for y in range(1, yy.shape[0]-1):
        for x in range(1, xx.shape[1]-1):
            neighborsNearProximity = []
            if norm(zz[y, x]) != 0:
                neighborsNearProximity.append(zz[y, x])
                p0Dist = norm(zz[y, x])
                if norm(zz[y-1, x-1]) != 0 and norm(zz[y-1, x-1] - zz[y, x]) < expectedRelPointDistanceUnderNoise*p0Dist:
                    neighborsNearProximity.append(zz[y-1, x-1])

                if norm(zz[y-1, x]) != 0 and norm(zz[y-1, x] - zz[y, x]) < expectedRelPointDistanceUnderNoise*p0Dist:
                    neighborsNearProximity.append(zz[y-1, x])

                if norm(zz[y+1, x+1]) != 0 and norm(zz[y+1, x+1] - zz[y, x]) < expectedRelPointDistanceUnderNoise*p0Dist:
                    neighborsNearProximity.append(zz[y+1, x+1])

                if norm(zz[y-1, x+1]) != 0 and norm(zz[y-1, x+1] - zz[y, x]) < expectedRelPointDistanceUnderNoise*p0Dist:
                    neighborsNearProximity.append(zz[y-1, x+1])

                if norm(zz[y, x-1]) != 0 and norm(zz[y, x-1] - zz[y, x]) < expectedRelPointDistanceUnderNoise*p0Dist:
                    neighborsNearProximity.append(zz[y, x-1])

                if norm(zz[y, x+1] - zz[y, x]) != 0 and norm(zz[y, x+1] - zz[y, x]) < expectedRelPointDistanceUnderNoise*p0Dist:
                    neighborsNearProximity.append(zz[y, x+1])

                if norm(zz[y+1, x-1]) != 0 and norm(zz[y+1, x-1] - zz[y, x]) < expectedRelPointDistanceUnderNoise*p0Dist:
                    neighborsNearProximity.append(zz[y+1, x-1])

                if norm(zz[y+1, x]) != 0 and norm(zz[y+1, x] - zz[y, x]) < expectedRelPointDistanceUnderNoise*p0Dist:
                    neighborsNearProximity.append(zz[y+1, x])

                nNeighbors = len(neighborsNearProximity)
                if nNeighbors > 3:
                    zz[y, x] = np.mean(neighborsNearProximity, axis=0)


    t2 = time.time()
    print("loading PCL, reshaping data, mean filtering: ", t2-t1)
    t1 = time.time()

    N = 1
    neighborhoodWndX = np.arange(-N,N+1)
    neighborhoodWndY = np.copy(neighborhoodWndX)
    neighborhoodWndXX, neighborhoodWndYY = np.meshgrid(neighborhoodWndX, neighborhoodWndY)
    numberNeighborsTotal = 4*2*N

    localNormals = np.zeros(shape=(xx.shape[0],xx.shape[1], numberNeighborsTotal+1, 2, 3))

    for y in range(N,yy.shape[0] - N):
        for x in range(N,xx.shape[1] - N):

            p0 = zz[y, x]
            p0Dist = norm(p0)
            if p0Dist != 0:
                avgVecNorm=np.array((0,0,0), dtype=np.float64)
                pNeighbors = []
                nValidNeighbors = 0
                upper = []
                right = []
                left = []
                lower = []
                for xOff in np.arange(-N,N+1):
                    upper.append(zz[y-N, x+xOff])
                    lower.append(zz[y+N, x+xOff])
                    left.append(zz[y-xOff, x-N])
                    right.append(zz[y+xOff, x+N])
                pNeighbors.append([upper, right, lower, left])

                for pNeighborhood in pNeighbors:
                    for pNeighborSeq in pNeighborhood:
                        for i in range(len(pNeighborSeq)-1):

                            if norm(pNeighborSeq[i]) != 0 and norm(p0-pNeighborSeq[i]) < expectedRelPointDistanceUnderNoise * p0Dist:
                                normalVec = cross(p0 - pNeighborSeq[i], p0 - pNeighborSeq[i+1])
                                nValidNeighbors += 1
                                normalVecNorm = normalVec / norm(normalVec)
                                avgVecNorm += normalVecNorm

                if nValidNeighbors > 3:
                    avgVecNorm = avgVecNorm / norm(avgVecNorm)
                    localNormals[y,x,numberNeighborsTotal,0]=p0
                    localNormals[y,x,numberNeighborsTotal,1]=avgVecNorm

    t2 = time.time()
    print("time for Normal calculation:", t2-t1)
    t1 = time.time()

    N = 1
    neighborhoodWndX = np.arange(-N, N+1)
    neighborhoodWndY = np.copy(neighborhoodWndX)
    neighborhoodWndXX, neighborhoodWndYY = np.meshgrid(neighborhoodWndX, neighborhoodWndY)
    numberNeighborsTotalNew = len(neighborhoodWndX)**2 - 1
    curvImg = np.zeros(shape=(xx.shape[0], xx.shape[1]))
    for y in range(N, yy.shape[0]-N):
        for x in range(N, xx.shape[1]-N):
            localCurv = 0
            n0 = localNormals[y, x, numberNeighborsTotal, 1]
            if norm(n0) == 0:
                curvImg[y, x] = 2
            else:
                nValidNeighbors = numberNeighborsTotalNew
                for nx in neighborhoodWndX:
                    for ny in neighborhoodWndY:
                        if nx == 0 and ny == 0:
                            continue
                        nNeighbor = localNormals[y+ny, x+nx, numberNeighborsTotal, 1]
                        if norm(nNeighbor) == 0:
                            nValidNeighbors -= 1
                        else:
                            localCurv += dot(n0, nNeighbor)
                if nValidNeighbors > 1:
                    localCurv /= nValidNeighbors
                    curvImg[y, x] = 1-localCurv
                else:
                    curvImg[y, x] = 2


    t2 = time.time()
    print("time for curvature calculation:", t2-t1)
    t1 = time.time()

    # filter measurueCurv threshold
    binCurvImg = binaCurvImg(curvImg, threshold=0.03)

    #binary closing
    binCurvImgInv = binCurvImg - 1
    binCurvImgInv[np.where(binCurvImgInv == -1)] = 1
    binCurvImgInvMorph = morph.binary_closing(binCurvImgInv, structure=np.ones((3,3))).astype(binCurvImgInv.dtype)
    binCurvImg = binCurvImgInvMorph-1
    binCurvImg[np.where(binCurvImg == -1)] = 1
    binCurvImg[:,:1] = 0
    binCurvImg[:1,:] = 0
    binCurvImg[binCurvImg.shape[0]-1:,:] = 0
    binCurvImg[:,binCurvImg.shape[1]-1:binCurvImg.shape[1]] = 0

    t2 = time.time()
    print("time for binarisation:", t2-t1)
    t1 = time.time()

    segments = segmentBinCurvImg(binCurvImg,0.05)
    segmentSize = [len(s) for s in segments]
    segmentedImg = np.zeros(shape=binCurvImg.shape)
    if len(segments) > 0:
        for segment in segments:
            for p in segment:
                segmentedImg[p[0], p[1]] = 1

    t2 = time.time()
    print("number of segements found: {}\nsegment sizes: {}\ntime for pre segmentation: {}".format(len(segments), segmentSize, t2-t1))
    t1 = time.time()

    filteredSegments, planes = filterOutlierSegments(segments, localNormals)
    segmentSize = [len(s) for s in filteredSegments]
    filteredSegmentedImg = np.zeros(shape=binCurvImg.shape)
    if len(filteredSegments) > 0:
        for segment in filteredSegments:
            for p in segment:
                filteredSegmentedImg[p[0], p[1]] = 1

    t2 = time.time()
    print("size of subsegements: {}\nsegment sizes: {}\ntime for model checking of segments: {}".format(len(filteredSegments), segmentSize, t2-t1))
    t1 = time.time()

    #expand to real pcl
    if len(filteredSegments)>0:
        N = 3
        neighborhoodWndX = np.arange(-N, N+1)
        neighborhoodWndY = np.copy(neighborhoodWndX)
        neighborhoodWndXX, neighborhoodWndYY = np.meshgrid(neighborhoodWndX, neighborhoodWndY)
        numberNeighborsTotalNew = len(neighborhoodWndX)**2 - 1

        zzExpanded = np.zeros(zzOrig.shape)
        planesBin = np.zeros((240, 320), dtype=np.uint8)
        for i, segment in enumerate(filteredSegments):
            for p in segment:
                p0 = (4 * p[0], 4 * p[1])
                if p[0] > 0 and p[1] > 0:
                    for nx in neighborhoodWndX:
                        for ny in neighborhoodWndY:

                            p = np.array(p0)+np.array((ny, nx))
                            tooDistant = checkDistanceToPlane(zzOrig[p[0], p[1]], planes[i],threshold=0.02)
                            if not tooDistant:
                                zzExpanded[p[0], p[1]] = zzOrig[p[0], p[1]]
                                planesBin[p[0], p[1]] = 255
                                
        t2 = time.time()
        print("PCL expansion: {}".format(t2-t1))
        t1 = time.time()
        zzSave = np.reshape(zzExpanded, (zzExpanded.shape[0] * zzExpanded.shape[1], 3))
        np.save(outputPathWithFileBasename + '_planesPCL', zzSave)
        cv2.imwrite(outputPathWithFileBasename + 'planesPCLMask.png', planesBin)
        cv2.imwrite(outputPathWithFileBasename + 'filteredSegmentedImg.png', filteredSegmentedImg * 255)
        
    cv2.imwrite(outputPathWithFileBasename + 'segmentedImg.png', segmentedImg * 255)
    cv2.imwrite(outputPathWithFileBasename + 'curvImgBin.png', binCurvImg * 255)
    cv2.imwrite(outputPathWithFileBasename + 'curvImg.png', curvImg * 255)

    t2 = time.time()
    print("Saving data: {}".format(t2-t1))
    
    planesDiscoveredAndLabeled = len(filteredSegments)>0
    
    return planesDiscoveredAndLabeled

if __name__=='__main__':
    dirName = '/media/gas/Samsung_T5/schlecht/'

    pcl_list = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        for file in filenames:
            if file.endswith('pcl_final.npy'):
                pcl_list.append((os.path.join(dirpath, file), os.path.join(dirpath, file)[:-13]))
    print(len(pcl_list))

    for pcl in pcl_list:
        labelPlanes(*pcl)

