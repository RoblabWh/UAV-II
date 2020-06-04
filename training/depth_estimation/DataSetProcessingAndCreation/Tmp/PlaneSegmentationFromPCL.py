# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 09:49:04 2019

@author: Marc, Gerhard

from mpl_toolkits import mplot3d as m3d
import matplotlib.pyplot as plt

...
theta=np.arange(0,2*np.pi,2*np.pi/5)
gamma=np.arange(0,2*np.pi,2*np.pi/5)

pth = []

for th in theta:
    for ga in gamma:
        x=np.cos(th)*np.sin(ga)
        y=np.sin(th)*np.sin(ga)
        z=np.cos(ga)
        pth.append([int(6+np.round(alpha*x)), int(5+np.round(alpha*y)),np.round(alpha*z)])
        
for p in pth:
    zz[p[1],p[0]]=p[2] if p[2] >= 0 else 0

ax = m3d.Axes3D(plt.figure())
ax.plot_surface(xx,yy,zz)
plt.show()
"""
import math

import numpy as np
from mpl_toolkits import mplot3d as m3d
import matplotlib.pyplot as plt
import time
from numba import jit


@jit(nopython=True, cache=True)
def cross(a, b):
    c = np.ones(3)
    c[0] = a[1] * b[2] - a[2] * b[1]
    c[1] = a[2] * b[0] - a[0] * b[2]
    c[2] = a[0] * b[1] - a[1] * b[0]
    return c

@jit(nopython=True, cache=True)
def norm(vec):
    return math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)

@jit(nopython=True, cache=True)
def dot(a, b):
     return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def getSimPCLFromObject(resx,resy,objectType='plane', step=False):
    
    #320,240
    x=np.arange(0,resx,dtype=int)
    y=np.arange(0,resy,dtype=int)
    
    xx,yy=np.meshgrid(x,y)
    
    zz=None
    if objectType=='plane':
        zz=np.ones((xx.shape))
    elif objectType=='noisyPlane':#
        zz = np.ones((xx.shape))
        gss = np.random.normal(size=zz.shape)
        zzNoisyPlane = zz * gss * 0.1
        return xx, yy, zzNoisyPlane
    elif objectType=='hemisphere':
        alpha = 0.25*np.min((resx,resy))
        alphaSquared=alpha**2
        zz=np.ones(xx.shape)
        zz=np.sqrt(np.abs(alphaSquared-(xx-0.5*resx)**2-(yy-0.5*resy)**2))
        zz[np.where(np.sqrt((xx-0.5*resx)**2+(yy-0.5*resy)**2) > alpha)] = 0
        return xx, yy, zz
    else:
        raise Exception("Object Type {} not yet supported! Goodbye".format(objectType))


xx, yy, zz = getSimPCLFromObject(320, 240, 'hemisphere')

t1=time.time()

# zzNoisyPlane=zz*gss*0.1
# zzNoisyPlaneStep=np.copy(zzNoisyPlane)
# zzNoisyPlaneStep[10:20 , 10:20] += 10

x_t=xx.reshape((xx.shape[0] * xx.shape[1]))
y_t=yy.reshape((yy.shape[0] * yy.shape[1]))
z_t=zz.reshape((zz.shape[0] * zz.shape[1]))
#zNoisyPlane = zzNoisyPlane.reshape((zzNoisyPlane.shape[0] * zzNoisyPlane.shape[1]))
#zNoisyPlaneStep = zz.reshape((zz.shape[0] * zz.shape[1]))

localNormals = np.zeros(shape=(xx.shape[0],xx.shape[1], 9, 2, 3))

for y in range(1,yy.shape[0]-1):
    for x in range(1,xx.shape[1]-1):
        p0=np.array((x, y, zz[y, x]))
        pNeighbors=[
                np.array((x, y - 1, zz[y - 1, x])),
                np.array((x + 1, y - 1, zz[y - 1, x + 1])),
                np.array((x + 1, y, zz[y, x + 1])),
                np.array((x + 1, y + 1, zz[y + 1, x + 1])),
                np.array((x, y + 1, zz[y + 1, x])),
                np.array((x - 1, y + 1, zz[y + 1, x - 1])),
                np.array((x - 1, y, zz[y, x - 1])),
                np.array((x - 1, y - 1, zz[y - 1, x - 1])),
                np.array((x, y - 1, zz[y - 1, x]))]
        
        avgVecNorm=np.array((0,0,0), dtype=np.float64)
        for i in range(len(pNeighbors)-1):
            
            normalVec = cross(p0 - pNeighbors[i], p0 - pNeighbors[i+1])
            normalVecNorm = normalVec / norm(normalVec)
            avgVecNorm += normalVecNorm
            
            triCenter = p0+pNeighbors[i]+pNeighbors[i+1]
            triCenter = triCenter / 3
            localNormals[y,x,i,0] = triCenter
            localNormals[y,x,i,1] = normalVecNorm
                  
        avgVecNorm = avgVecNorm / (len(pNeighbors)-1)
        localNormals[y,x,8,0]=p0
        localNormals[y,x,8,1]=avgVecNorm

measureCurv = np.zeros(shape=(xx.shape[0],xx.shape[1]))
for y in range(2,yy.shape[0]-2):
    for x in range(2,xx.shape[1]-2):
        localCurv = 0
        n0=localNormals[y,x,8,1]
        nNeighbors=[
                localNormals[y-1,x,8,1],
                localNormals[y-1,x+1,8,1],
                localNormals[y,x+1,8,1],
                localNormals[y+1,x+1,8,1],
                localNormals[y+1,x,8,1],
                localNormals[y+1,x-1,8,1],
                localNormals[y,x-1,8,1],
                localNormals[y-1,x-1,8,1]]
        for nNeighbor in nNeighbors:
            localCurv += dot(n0, nNeighbor)
        localCurv /= 8
        measureCurv[y,x] = localCurv

t2=time.time()
print("time for calculation:", t2-t1)


# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(x_t, y_t, z_t, s=5) #, color=(0,0,1,0)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_xlim3d(0, 300)
# ax.set_ylim3d(0,300)
# ax.set_zlim3d(0, 300)
#
# x=localNormals[:,:,:8,0,0]
# y=localNormals[:,:,:8,0,1]
# z=localNormals[:,:,:8,0,2]
# u,v,w = (localNormals[:,:,:8,1,0], localNormals[:,:,:8,1,1], localNormals[:,:,:8,1,2])
#
# #ax.quiver(x, y, z, u, v, w, length=0.5, color=(0,1,0,1)) #, color=(0,1,0,0)
#
# x=localNormals[:,:,8,0,0]
# y=localNormals[:,:,8,0,1]
# z=localNormals[:,:,8,0,2]
# u,v,w = (localNormals[:,:,8,1,0], localNormals[:,:,8,1,1], localNormals[:,:,8,1,2])
#
# colors=[(1,0,0,1) for e in x]
# #ax.quiver(x, y, z, u, v, w, length=1, color=(1,0,0,1)) #, color=(1,0,0,0)
#
# #plt.show()


#plt.contourf(xx,yy,measureCurv, cmap='hot')
plt.matshow(measureCurv, cmap='hot')
plt.colorbar()
plt.show()