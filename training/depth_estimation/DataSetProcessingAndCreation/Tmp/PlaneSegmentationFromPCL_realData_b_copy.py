# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 09:49:04 2019
​
@author: Marc, Gerhard

"""
import math
import numpy as np
from mpl_toolkits import mplot3d as m3d
import matplotlib.pyplot as plt
import time
from numba import jit
import scipy.ndimage.morphology as morph
from sklearn import linear_model
import random
import copy


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


def getSimPCLFromObject(resx, resy, objectType='plane', step=False):
    
    #320,240
    x = np.arange(0, resx, dtype=int)
    y = np.arange(0, resy, dtype=int)
    
    xx, yy = np.meshgrid(x, y)
    
    zz = None
    if objectType == 'plane':
        zz = np.ones((xx.shape))
    elif objectType == 'noisyPlane':#
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
        #yy = (yy + gss*0.5)
        #xx = (xx + gss*0.5)
    else:
        raise Exception("Object Type {} not yet supported! Goodbye".format(objectType))
        
    zz2 = np.zeros(shape=(xx.shape[0], xx.shape[1], 3))
    for r in range(xx.shape[0]):
        for c in range(xx.shape[1]):  
            zz2[r, c] = np.array((xx[r, c], yy[r, c], zz[r, c]))
            
    return xx,yy,zz2



def binCurvImg(curvImg, threshold=0.03):
    
    measureCurvBin = np.zeros(shape=curvImg.shape,dtype=int)
    for y in range(yy.shape[0]):
        for x in range(xx.shape[1]):
            if np.abs(curvImg[y,x]) < threshold:
                measureCurvBin[y,x] = 1
                    
    return measureCurvBin

def segmentBinCurvImg(binCurvImg, minSegmentSizeRel=0.1):
    
    segments = []
    
    thesisSet = set()
    for r,row in enumerate(binCurvImg):
        for p, binPixel in enumerate(row):
            if binPixel==1:
                thesisSet.add((r,p))
    
    while len(thesisSet) > 0:
        primeSeed = thesisSet.pop()
        regionSet = set()
        workingSet = set()
        regionSet.add(primeSeed)
        workingSet.add(primeSeed)
        while len(workingSet) > 0:
            currentSeed = workingSet.pop()
            v,u = currentSeed
            if  binCurvImg.shape[1]-1 > u >= 0:
                if binCurvImg[v, u+1]==1:
                    right = (currentSeed[0], currentSeed[1]+1)
                    if right not in regionSet:
                        workingSet.add(right)
                        regionSet.add(right)
            if binCurvImg.shape[1] >= u > 1:
                if binCurvImg[v, u-1]==1:
                    left = (currentSeed[0], currentSeed[1]-1)
                    if left not in regionSet:
                        regionSet.add(left)
                        workingSet.add(left)
            if binCurvImg.shape[0] >= v > 1:
                if binCurvImg[v-1, u]==1:
                    up = (currentSeed[0]-1, currentSeed[1])
                    if up not in regionSet:
                        regionSet.add(up)
                        workingSet.add(up)
            if binCurvImg.shape[0]-1 > v >= 0:
                if binCurvImg[v+1, u]==1:
                    down = (currentSeed[0]+1, currentSeed[1])
                    if down not in regionSet:
                        regionSet.add(down)
                        workingSet.add(down)
        
        thesisSet = thesisSet - regionSet
        imgSize = binCurvImg.shape[0]*binCurvImg.shape[1]
        if len(regionSet) >= minSegmentSizeRel * imgSize:
            minSegmentSizeRel = 0.5*len(regionSet) / imgSize
            segments.append(regionSet)
    
    return segments

def getThreeRandomPoints(points):

    #randomIndices = random.sample(range(0, len(points)), 3)
    #return points[randomIndices[0]], points[randomIndices[1]], points[randomIndices[2]]
    
    p1Index = random.randint(0, len(points)-1)
    p2Index = random.randint(0, len(points)-1)  
    while p2Index == p1Index:
        p2Index = random.randint(0, len(points)-1)
        
    p3Index = random.randint(0, len(points)-1)
    while p3Index == p2Index or p3Index == p1Index:
        p3Index = random.randint(0, len(points)-1)
    
    p1 = points[p1Index]
    p2 = points[p2Index]
    p3 = points[p3Index]
    
    return p1,p2,p3

def ransacPlane(segment, localNormals):
    
    filteredSegment = []
    
    superSegment3D = []
    superSegment2D3D = []
    for s in segment:
        superSegment2D3D.append((s, localNormals[s[0],s[1],8,0]))
        superSegment3D.append(localNormals[s[0],s[1],8,0])
        
    outlier_q = 0.2
    p = 0.99
    epsilon = 0.02
    k = np.log(1-p)/np.log(1-(1-outlier_q)**3)
    k = int(np.ceil(k))
    
    bestConsensus = None
    consensusSet = None
    for i in range(k):
        p1,p2,p3 = getThreeRandomPoints(superSegment3D)
        normal = np.cross(np.array(p2)-np.array(p1), np.array(p3) - np.array(p1))
        normal = normal / norm(normal)
        dPlane = np.dot(normal, p1)
        #plane = {'plane distance' : d, 'plane orientation' : normal}
        consensusSet = set()
        for pTupel in superSegment2D3D:
            p = pTupel[1]
            d = np.abs(dot(normal, np.array(p)) - dPlane)
            if d < epsilon:
                pHashable = ((pTupel[0][0], pTupel[0][1]), (pTupel[1][0], pTupel[1][1], pTupel[1][2]))
                consensusSet.add(pHashable)
                
                
        if bestConsensus is None:
            bestConsensus = copy.deepcopy(consensusSet)
            print(len(bestConsensus))
        else:
            if len(consensusSet) > len(bestConsensus):
                print("new: ", len(consensusSet),len(bestConsensus))
                bestConsensus = copy.deepcopy(consensusSet)
                
    filteredSegment = list(bestConsensus)
    print("finally: ", len(bestConsensus),len(filteredSegment))
    
    return filteredSegment
    
def filterOutlierSegments(segments, localNormals):
    
    filteredSegments = []
    
    for segment in segments:
        filteredSegment = ransacPlane(segment, localNormals)
        filteredSegmentImg = np.zeros(shape=(localNormals.shape[0], localNormals.shape[1]))
        for fs in filteredSegment:
            pCoo = fs[0]
            filteredSegmentImg[pCoo[0], pCoo[1]] = 1
            
        filteredSegmentSub = segmentBinCurvImg(filteredSegmentImg, minSegmentSizeRel=0.1)
        if len(filteredSegmentSub) > 0:
            filteredSegments.append(filteredSegmentSub[0])

    return filteredSegments

t1=time.time()    
#xx, yy, zz = getSimPCLFromObject(640, 480, 'NoisyHemisphere')


#pcl = np.load(r'E:\transformedNYUDataset\basement_0001a\rgb1316653580_471513_rgbd1316653580_484909\d-1316653580.471513-1316138413pclKinect.npy')
#pcl=np.load(r'E:\transformedNYUDataset\basement_0001a\rgb1316653587_489233_rgbd1316653587_512883\d-1316653587.489233-1736590963pclKinect.npy')
#pcl = np.load(r'E:\transformedNYUDataset\basement_0001a\rgb1316653585_885734_rgbd1316653585_912940\d-1316653585.885734-1640487523pclKinect.npy')
#pcl = np.load(r'E:\transformedNYUDataset_b\basement_0001b\rgb1316653649_123522_rgbd1316653649_152430\d-1316653649.123522-1139603952pclKinect.npy')
#pcl = np.load(r'E:\transformedNYUDataset_LivingRoom\living_room_0002\rgb1294890235_349177_rgbd1294890235_374729\d-1294890235.349177-3030778186pclKinect.npy')
#pcl = np.load(r'E:\transformedNYUDataset_LivingRoom\living_room_0005\rgb1295148616_605279_rgbd1295148616_642831\d-1295148616.605279-1131612824pclKinect.npy')
#pcl = np.load(r'E:\transformedNYUDataset_LivingRoom\living_room_0010\rgb1295836456_718805_rgbd1295836456_724393\d-1295836456.718805-1139633710pclKinect.npy')
pcl = np.load(r'/home/gas/Dokumente/PlaneSegmentation/PlanesFromPCL/Nine/d-1315418961.223898-3111310978pclKinect.npy')
#pcl = np.load(r'/home/gas/Dokumente/PlaneSegmentation/PlanesFromPCL/First/d-1316653580.471513-1316138413pclKinect.npy')

x = np.arange(0,640,dtype=int)
y = np.arange(0,480,dtype=int)
xx,yy=np.meshgrid(x,y)
zzOrig=np.zeros(shape=(xx.shape[0],xx.shape[1],3))
for i,row in enumerate(zzOrig):
    for j,p in enumerate(row):
        index = i*zzOrig.shape[1]+int(j)
        zzOrig[i,int(j)]=pcl[index]

xx=xx[::8,::8]
yy=yy[::8,::8]
zz=zzOrig[::8,::8]

for y in range(1,yy.shape[0]-1):
    for x in range(1,xx.shape[1]-1):
        neighborsNearProximity = []
        if norm(zz[y,x]) != 0:
            neighborsNearProximity.append(zz[y,x])
            p0Dist = norm(zz[y,x])
            if norm(zz[y-1,x-1]) != 0 and norm(zz[y-1,x-1]-zz[y,x]) < expectedRelPointDistanceUnderNoise*p0Dist:
                neighborsNearProximity.append(zz[y-1,x-1])
                
            if norm(zz[y-1,x]) != 0 and norm(zz[y-1,x]-zz[y,x]) < expectedRelPointDistanceUnderNoise*p0Dist:
                neighborsNearProximity.append(zz[y-1,x])
            
            if norm(zz[y+1,x+1]) != 0 and norm(zz[y+1,x+1]-zz[y,x]) < expectedRelPointDistanceUnderNoise*p0Dist:
                neighborsNearProximity.append(zz[y+1,x+1])
            
            if norm(zz[y-1,x+1]) != 0 and norm(zz[y-1,x+1]-zz[y,x]) < expectedRelPointDistanceUnderNoise*p0Dist:
                neighborsNearProximity.append(zz[y-1,x+1])
            
            if norm(zz[y,x-1]) != 0 and norm(zz[y,x-1]-zz[y,x]) < expectedRelPointDistanceUnderNoise*p0Dist:
                neighborsNearProximity.append(zz[y,x-1])
                
            if norm(zz[y,x+1]-zz[y,x]) != 0 and norm(zz[y,x+1]-zz[y,x]) < expectedRelPointDistanceUnderNoise*p0Dist:
                neighborsNearProximity.append(zz[y,x+1])
            
            if norm(zz[y+1,x-1]) != 0 and norm(zz[y+1,x-1]-zz[y,x]) < expectedRelPointDistanceUnderNoise*p0Dist:
                neighborsNearProximity.append(zz[y+1,x-1])
                
            if norm(zz[y+1,x]) != 0 and norm(zz[y+1,x]-zz[y,x]) < expectedRelPointDistanceUnderNoise*p0Dist:
                neighborsNearProximity.append(zz[y+1,x])
            
            nNeighbors = len(neighborsNearProximity)
            if nNeighbors > 3:
                zz[y, x] = np.mean(neighborsNearProximity, axis=0)



t2=time.time()
print("loading PCL, reshaping data, mean filtering: ", t2-t1)
t1=time.time()

N=1
neighborhoodWndX = np.arange(-N,N+1)
neighborhoodWndY = np.copy(neighborhoodWndX)
neighborhoodWndXX,neighborhoodWndYY=np.meshgrid(neighborhoodWndX,neighborhoodWndY)
numberNeighborsTotal = 4*2*N

localNormals = np.zeros(shape=(xx.shape[0],xx.shape[1], numberNeighborsTotal+1, 2, 3))

for y in range(N,yy.shape[0] - N):
    for x in range(N,xx.shape[1] - N):
        
        p0=zz[y, x]
        p0Dist = norm(p0)
        if p0Dist!=0:
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
                            #triCenter = p0+pNeighborSeq[i]+pNeighborSeq[i+1]
                            #triCenter = triCenter / 3
                            #localNormals[y,x,i,0] = triCenter
                            #localNormals[y,x,i,1] = normalVecNorm
                
            if nValidNeighbors > 3:
                avgVecNorm = avgVecNorm / norm(avgVecNorm)
                localNormals[y,x,numberNeighborsTotal,0]=p0
                localNormals[y,x,numberNeighborsTotal,1]=avgVecNorm
            
t2=time.time()
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
                    nNeighbor=localNormals[y+ny,x+nx,numberNeighborsTotal,1]
                    if norm(nNeighbor) == 0:
                        nValidNeighbors -= 1
                    else:
                        localCurv += dot(n0, nNeighbor)
            if nValidNeighbors > 1:
                localCurv /= nValidNeighbors
                curvImg[y,x] = 1-localCurv
            else:
                curvImg[y,x] = 2
     

t2=time.time()
print("time for curvature calculation:", t2-t1)
t1 = time.time()

#filter measurueCurv threshold

binCurvImg = binCurvImg(curvImg,threshold=0.005)

#binary closing
binCurvImgInv = binCurvImg-1
binCurvImgInv[np.where(binCurvImgInv == -1)] = 1
binCurvImgInvMorph = morph.binary_closing(binCurvImgInv, structure=np.ones((3,3))).astype(binCurvImgInv.dtype)
binCurvImg=binCurvImgInvMorph-1
binCurvImg[np.where(binCurvImg == -1)]=1
binCurvImg[:,:1]=0
binCurvImg[:1,:]=0
binCurvImg[binCurvImg.shape[0]-1:,:] = 0
binCurvImg[:,binCurvImg.shape[1]-1:binCurvImg.shape[1]] = 0

t2=time.time()
print("time for binarisation:", t2-t1)
t1=time.time()

segments = segmentBinCurvImg(binCurvImg)
segmentSize = [len(s) for s in segments]
segmentedImg = np.zeros(shape=binCurvImg.shape)
if len(segments)>0:
    for p in segments[0]:
        segmentedImg[p[0],p[1]] = 1

t2=time.time()
print("number of segements found: {}\nsegment sizes: {}\ntime for pre segmentation: {}".format(len(segments), segmentSize, t2-t1))
t1=time.time()

filteredSegments = filterOutlierSegments(segments, localNormals)
segmentSize = [len(s) for s in filteredSegments]
filteredSegmentedImg = np.zeros(shape=binCurvImg.shape)
if len(filteredSegments)>0:
    for p in filteredSegments[0]:
        filteredSegmentedImg[p[0], p[1]] = 1
        
t2=time.time()
print("size of subsegements: {}\nsegment sizes: {}\ntime for model checking of segments: {}".format(len(filteredSegments), segmentSize, t2-t1))
t1=time.time()

#expand to real pcl
bestBetSegment = filteredSegments[0]
N = 3
neighborhoodWndX = np.arange(-N, N+1)
neighborhoodWndY = np.copy(neighborhoodWndX)
neighborhoodWndXX, neighborhoodWndYY=np.meshgrid(neighborhoodWndX, neighborhoodWndY)
numberNeighborsTotalNew = len(neighborhoodWndX)**2 - 1
zzExpanded = np.zeros(shape=zzOrig.shape)
for p in filteredSegments[0]:
    p0 = (8 * p[0], 8 * p[1])
    zzExpanded[p0[0], p0[1]] = zzOrig[p0[0], p0[1]]
    for nx in neighborhoodWndX:
        for ny in neighborhoodWndY:
            if nx == 0 and ny == 0:
                continue
            p = np.array(p0)+np.array((ny, nx))
            #print(p[0],p[1], zzOrig[p[0],p[1]])
            zzExpanded[p[0], p[1]] = zzOrig[p[0], p[1]]

zzSave = np.reshape(zzExpanded, (zzExpanded.shape[0] * zzExpanded.shape[1], 3))
np.save(r'C:\Users\Marc\Desktop\first\firstPCLSegment', zzSave)
#plt.figure(111)
plt.matshow(curvImg, cmap='hot')
plt.colorbar()

plt.figure(211)
plt.imshow(binCurvImg, cmap=plt.cm.gray)

plt.figure(311)
plt.imshow(segmentedImg, cmap=plt.cm.gray)

plt.figure(411)
plt.imshow(filteredSegmentedImg, cmap=plt.cm.gray)


plt.show()
#plt.figure(311)
#plt.imshow(measureCurvBin ,cmap=plt.cm.gray)
#plt.show()

'''
fig = plt.figure(211)
ax = plt.axes(projection='3d')
xx = np.array([[p[0] for p in row] for row in zz])
yy = np.array([[p[1] for p in row] for row in zz])
zz = np.array([[p[2] for p in row] for row in zz])
x_t=xx.reshape((xx.shape[0] * xx.shape[1]))
y_t=yy.reshape((yy.shape[0] * yy.shape[1]))
z_t=zz.reshape((zz.shape[0] * zz.shape[1]))

ax.scatter3D(x_t, y_t, z_t, s=5) #, color=(0,0,1,0)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_xlim3d(0, 320)
ax.set_ylim3d(0,320)
ax.set_zlim3d(100, 380)
plt.show()
'''