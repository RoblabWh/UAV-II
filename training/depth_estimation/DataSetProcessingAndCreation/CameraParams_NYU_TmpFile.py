# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 16:42:41 2019

@author: Marc
"""

# Calibrated using the RGBDemo Calibration tool:
#   http://labs.manctl.com/rgbdemo/
# 
import numpy as np

# The maximum depth used, in meters.
maxDepth = 10

# RGB Intrinsic Parameters
fx_rgb = 5.1885790117450188e+02
fy_rgb = 5.1946961112127485e+02
cx_rgb = 3.2558244941119034e+02
cy_rgb = 2.5373616633400465e+02

# RGB Distortion Parameters
k1_rgb =  2.0796615318809061e-01
k2_rgb = -5.8613825163911781e-01
p1_rgb = 7.2231363135888329e-04
p2_rgb = 1.0479627195765181e-03
k3_rgb = 4.9856986684705107e-01

# Depth Intrinsic Parameters
fx_d = 5.8262448167737955e+02
fy_d = 5.8269103270988637e+02
cx_d = 3.1304475870804731e+02
cy_d = 2.3844389626620386e+02

# RGB Distortion Parameters
k1_d = -9.9897236553084481e-02
k2_d = 3.9065324602765344e-01
p1_d = 1.9290592870229277e-03
p2_d = -1.9422022475975055e-03
k3_d = -5.1031725053400578e-01

# Rotation
R = 1 * np.array([ 9.9997798940829263e-01, 5.0518419386157446e-03, 4.3011152014118693e-03, -5.0359919480810989e-03, 9.9998051861143999e-01, -3.6879781309514218e-03,-4.3196624923060242e-03, 3.6662365748484798e-03, 9.9998394948385538e-01 ])

R = np.reshape(R, (3,3))
R = np.transpose(R)

# 3D Translation
t_x = 2.5031875059141302e-02
t_z = -2.9342312935846411e-04
t_y = 6.6238747008330102e-04

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
    
    