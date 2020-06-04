# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 16:42:41 2019

@author: Marc Thurow
"""

import numpy as np

# The maximum depth used, in meters.
maxDepth = 10
#maxDepth = 10000 #mm

# RGB Intrinsic Parameters
fx_rgb = 5.19206635e+02
fy_rgb = 5.18724844e+02
cx_rgb = 3.26432036e+02
cy_rgb = 2.44559453e+02

# RGB Distortion Parameters
k1_rgb =  1.049413e-01
k2_rgb = 7.587888e-02
p1_rgb = -2.204494e-03
p2_rgb = 8.126022e-04
k3_rgb = -6.905027e-01

# RGB Distortion Parameters
k1_rgb =  2.0796615318809061e-01
k2_rgb = -5.8613825163911781e-01
p1_rgb = 7.2231363135888329e-04
p2_rgb = 1.0479627195765181e-03
k3_rgb = 4.9856986684705107e-01

# RGB Distortion Parameters
k1_rgb =  0
k2_rgb = 0
p1_rgb = 0
p2_rgb = 0
k3_rgb = 0

# Depth Intrinsic Parameters
fx_d = 5.73459978e+02
fy_d = 5.72966907e+02
cx_d = 3.20366078e+02
cy_d = 2.38576996e+02

# Depth Distortion Parameters
k1_d = -7.703475e-03
k2_d = -2.533582e-01
p1_d = 2.784642e-03
p2_d = 1.788255e-04
k3_d = 6.620651e-01

# RGB Distortion Parameters
k1_d = -9.9897236553084481e-02
k2_d = 3.9065324602765344e-01
p1_d = 1.9290592870229277e-03
p2_d = -1.9422022475975055e-03
k3_d = -5.1031725053400578e-01

# RGB Distortion Parameters
k1_d = 0
k2_d = 0
p1_d = 0
p2_d = 0
k3_d = 0

# Rotation
R = 1 * np.array([ 9.9997880829255e-01, 6.2032311957821e-03, 1.9755729632565e-03, -6.2055066354864e-03, 9.9998008698295e-01, 1.1477477743765e-03, -1.9684138788395e-03, -1.1599828827738e-03, 9.9999738988985e-01 ])

R = np.reshape(R, (3,3))
R = np.transpose(R)

# 3D Translation
t_x = 2.445536e-02
t_z = -9.880685e-04
t_y = -9.833095e-03

# Parameters for making depth absolute.
depthParam1 = 351.3;
depthParam2 = 1092.5;

def getCameraParams():
    cameraParams = {'RGB' : None, 'RGBD' : None, 'DepthParams': None, 'RGBD_To_RGB' : None}
    
    '''
    rgbIntrinsics = {'fx': fx_rgb, 'fy': fy_rgb, 'cx': cx_rgb, 'cy': cy_rgb,
                     'k1': k1_rgb, 'k2': k2_rgb, 'p1': p1_rgb, 'p2': p2_rgb, 'k3': k3_rgb}
    '''
    rgbIntrinsics = {'pinhole': np.array( ((fx_rgb,0,cx_rgb),(0,fy_rgb,cy_rgb),(0,0,1)) ), 'distCoeffs': np.array((k1_rgb,k2_rgb,p1_rgb,p2_rgb,k3_rgb))}
    
    '''
    rgbdIntrinsics = {'fx':fx_d, 'fy': fy_d, 'cx': cx_d, 'cy': cy_d,
                      'k1': k1_d, 'k2': k2_d, 'p1': p1_d, 'p2': p2_d, 'k3': k3_d}
    '''
    rgbdIntrinsics = {'pinhole': np.array( ((fx_d,0,cx_d),(0,fy_d,cy_d),(0,0,1)) ), 'distCoeffs': np.array((k1_d,k2_d,p1_d,p2_d,k3_d))}
    rgbdDepthParams = {'maxDepth' : maxDepth, 'depthParam1': depthParam1, 'depthParam2': depthParam2}
    
    rgbdToRgbTransform = {'R': R, 't': [t_x, t_y, t_z]}
    
    cameraParams['RGB'] = rgbIntrinsics
    cameraParams['RGBD'] = rgbdIntrinsics
    cameraParams['DepthParams'] = rgbdDepthParams
    cameraParams['RGBD_To_RGB'] = rgbdToRgbTransform
    
    return cameraParams
    
    
