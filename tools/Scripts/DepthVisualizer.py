# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 16:39:13 2019

@author: Marc
"""

import sys
import numpy as np

from scipy import ndimage
from scipy.misc import imread
from skimage.transform import resize
import NYU_cameraParams


# UI and OpenGL
from PyQt5 import QtCore, QtGui, QtWidgets, QtOpenGL
from OpenGL import GL, GLU
from OpenGL.arrays import vbo
from OpenGL.GL import shaders
import glm


# Image shapes
height_rgb, width_rgb = 480, 640
height_depth, width_depth = height_rgb // 2, width_rgb // 2
rgb_width = width_rgb
rgb_height = height_rgb


# Conversion from Numpy to QImage and back
def np_to_qimage(a):
    im = a.copy()
    return QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_Grayscale8).copy()
    
def qimage_to_np(qimage, dtype = 'array'):
	"""Convert QImage to numpy.ndarray.  The dtype defaults to uint8
	for QImage.Format_Indexed8 or `bgra_dtype` (i.e. a record array)
	for 32bit color images.  You can pass a different dtype to use, or
	'array' to get a 3D uint8 array for color images."""
	result_shape = (qimage.height(), qimage.width())
	temp_shape = (qimage.height(),
				  qimage.bytesPerLine() * 8 // qimage.depth())
	if qimage.format() in (QtGui.QImage.Format_ARGB32_Premultiplied,
						   QtGui.QImage.Format_ARGB32,
						   QtGui.QImage.Format_RGB32):

		if dtype == 'array':
			dtype = np.uint8
			result_shape += (4, )
			temp_shape += (4, )
	elif qimage.format() == QtGui.QImage.Format_Indexed8:
		dtype = np.uint8
	else:
		raise ValueError("qimage2numpy only supports 32bit and 8bit images")
	# FIXME: raise error if alignment does not match
	buf = qimage.bits().asstring(qimage.byteCount())
	result = np.frombuffer(buf, dtype).reshape(temp_shape)
	if result_shape != temp_shape:
		result = result[:,:result_shape[1]]
	if qimage.format() == QtGui.QImage.Format_RGB32 and dtype == np.uint8:
		result = result[...,:3]
	return result

# Compute edge magnitudes
def edges(d):
    dx = ndimage.sobel(d, 0)  # horizontal derivative
    dy = ndimage.sobel(d, 1)  # vertical derivative
    return np.abs(dx) + np.abs(dy)

# Main window
class Window(QtWidgets.QWidget):

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)


        mainLayout = QtWidgets.QVBoxLayout()
        
         #  views
        viewsLayout = QtWidgets.QGridLayout()

        self.inputViewer = QtWidgets.QLabel("Kommt wenn du npy file lädst")
        self.inputViewer.setPixmap(QtGui.QPixmap(rgb_width,rgb_height))
        
        imgFrame = QtWidgets.QFrame()   
        rgbLayout = QtWidgets.QVBoxLayout()
        rgbLayout.addWidget(self.inputViewer)
        imgFrame.setLayout(rgbLayout)
        viewsLayout.addWidget(imgFrame,0,0)
        
        self.glWidget = GLWidget()
        viewsLayout.addWidget(self.glWidget,0,1)
        
        mainLayout.addLayout(viewsLayout)

        # Load depth estimation model      
        toolsLayout = QtWidgets.QHBoxLayout()  

        self.buttonPcl = QtWidgets.QPushButton("Load PGM...")
        self.buttonPcl.clicked.connect(self.loadImg)
        toolsLayout.addWidget(self.buttonPcl)
        
        self.buttonImg = QtWidgets.QPushButton("Load Image...")
        self.buttonImg.clicked.connect(self.loadColorImg)
        toolsLayout.addWidget(self.buttonImg)

        mainLayout.addLayout(toolsLayout)        

        self.setLayout(mainLayout)
        self.setWindowTitle(self.tr("PCL Viewer"))
    
    def loadImg(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(None, 'Select Depth Image', '', self.tr('PGM files (*.png)'))[0]
        img = imread(filename)
        
        self.inputViewer.setPixmap(QtGui.QPixmap.fromImage(np_to_qimage(img)))
        grey=img.astype('float32')
        grey2 = np.zeros(shape=(grey.shape[0],grey.shape[1],3),dtype='float32')
        grey2[:,:,0] = grey[:,:]
        grey2[:,:,1] = grey[:,:]
        grey2[:,:,2] = grey[:,:]
        grey = resize((grey[:,:]/255), (rgb_height,rgb_width), order=1)
        grey2 = resize(np.copy(grey2/255),(rgb_height*rgb_width,3), order=1)
        grey2=grey2.astype("float32")
        #self.glWidget.createImgVBO(grey2)
        
        cameraParams = NYU_cameraParams.getCameraParams()
        pcl = self.createPcl(grey, cameraParams)
        print(pcl)
        self.glWidget.createPointCloudVBO(pcl)
        
    def loadColorImg(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(None, 'Select Color Image', '', self.tr('PGM files (*.jpg)'))[0]
        img = imread(filename)
        
        rgb = resize((img[:,:,:3]/255)[:,:,::-1], (rgb_height, rgb_width), order=1)
        rgb=rgb.astype('float32')
        self.glWidget.createImgVBO(rgb)
        
        
    def createPcl(self, undistRgbd, cameraParams):
        
        #pcl = np.zeros(shape=(undistRgbd.shape[0], undistRgbd.shape[1], 3)) #480x640x3
        pcl = None
        undistRgbdMm = undistRgbd
        cameraMatrix = cameraParams['RGB']['pinhole']
        print(cameraMatrix)
        x = np.arange(0,undistRgbdMm.shape[1])
        y = np.arange(0,undistRgbdMm.shape[0])
        xx,yy = np.meshgrid(x, y)

        cx = cameraMatrix[0,2]
        cy = cameraMatrix[1,2]
        fx = cameraMatrix[0,0]
        fy = cameraMatrix[1,1]*1.2
        
        depth=undistRgbdMm
        X = (xx - cx) / fx * depth
        Y = (yy - cy) / fy * depth
        Z = depth
        #pcl=np.dstack((X, Y, Z)).reshape((width*height, 3))
        
        pcl = np.stack([X,Y,Z],axis=2)
        pcl=np.reshape(pcl, (undistRgbdMm.shape[1]*undistRgbdMm.shape[0],3))
        print(pcl[300:310])
        print("end")
        
        return pcl.astype('float32')
        
        

class GLWidget(QtOpenGL.QGLWidget):
    xRotationChanged = QtCore.pyqtSignal(int)
    yRotationChanged = QtCore.pyqtSignal(int)
    zRotationChanged = QtCore.pyqtSignal(int)
    def __init__(self, parent=None):
        QtOpenGL.QGLWidget.__init__(self, parent)

        self.object = 0
        self.xRot = 5040
        self.yRot = 40
        self.zRot = 0
        self.zoomLevel = 9

        self.lastPos = QtCore.QPoint()

        self.trolltechGreen = QtGui.QColor.fromCmykF(0.40, 0.0, 1.0, 0.0)
        self.trolltechPurple = QtGui.QColor.fromCmykF(0.39, 0.39, 0.0, 0.0)

        self.pos_vbo = None
        self.col_vbo = None
        self.rgb=None

    def xRotation(self):
        return self.xRot

    def yRotation(self):
        return self.yRot

    def zRotation(self):
        return self.zRot

    def minimumSizeHint(self):
        return QtCore.QSize(640, 480)

    def sizeHint(self):
        return QtCore.QSize(640, 480)

    def setXRotation(self, angle):
        if angle != self.xRot:
            self.xRot = angle
            #self.emit(QtCore.SIGNAL("xRotationChanged(int)"), angle)
            self.xRotationChanged.emit(angle)
            self.updateGL()

    def setYRotation(self, angle):
        if angle != self.yRot:
            self.yRot = angle
            #self.emit(QtCore.SIGNAL("yRotationChanged(int)"), angle)
            self.yRotationChanged.emit(angle)
            self.updateGL()

    def setZRotation(self, angle):
        if angle != self.zRot:
            self.zRot = angle
            #self.emit(QtCore.SIGNAL("zRotationChanged(int)"), angle)
            self.zRotationChanged.emit(angle)
            self.updateGL()

    def resizeGL(self, width, height):
        GL.glViewport(0, 0, width, height)            

    def mousePressEvent(self, event):
        self.lastPos = QtCore.QPoint(event.pos())

    def mouseMoveEvent(self, event):
        dx = -(event.x() - self.lastPos.x())
        dy = (event.y() - self.lastPos.y())

        if event.buttons() & QtCore.Qt.LeftButton:
            self.setXRotation(self.xRot + dy)
            self.setYRotation(self.yRot + dx)
        elif event.buttons() & QtCore.Qt.RightButton:
            self.setXRotation(self.xRot + dy)
            self.setZRotation(self.zRot + dx)

        self.lastPos = QtCore.QPoint(event.pos())

    def wheelEvent(self, event):
        numDegrees = event.angleDelta().y() / 2
        numSteps = numDegrees / 10
        self.zoomLevel = self.zoomLevel + numSteps
        event.accept()
        self.updateGL()

    def initializeGL(self):
        self.qglClearColor(self.trolltechPurple.darker())
        GL.glShadeModel(GL.GL_FLAT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)

        VERTEX_SHADER = shaders.compileShader("""#version 330
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 color;
        uniform mat4 mvp; out vec4 frag_color;
        void main() {gl_Position = mvp * vec4(position, 1.0);frag_color = vec4(color, 1.0);}""", GL.GL_VERTEX_SHADER)

        FRAGMENT_SHADER = shaders.compileShader("""#version 330
        in vec4 frag_color; out vec4 out_color; 
        void main() {out_color = frag_color;}""", GL.GL_FRAGMENT_SHADER)

        self.shaderProgram = shaders.compileProgram(VERTEX_SHADER, FRAGMENT_SHADER)
        
        self.UNIFORM_LOCATIONS = {
            'position': GL.glGetAttribLocation( self.shaderProgram, 'position' ),
            'color': GL.glGetAttribLocation( self.shaderProgram, 'color' ),
            'mvp': GL.glGetUniformLocation( self.shaderProgram, 'mvp' ),
        }

        shaders.glUseProgram(self.shaderProgram)

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        self.drawObject()

    def createPointCloudVBO(self,pcl):
        # Create position and color VBOs
        
        self.pos=np.copy(pcl)
        self.pos = self.pos + glm.vec3(0, -0.06, -0.3)
        self.pos_vbo = vbo.VBO(data=self.pos, usage=GL.GL_DYNAMIC_DRAW, target=GL.GL_ARRAY_BUFFER)
        #if not self.rgb is None:
            #self.createImgVBO(self.rgb)
    
    
    def createImgVBO(self,img):
        self.rgb=img
        self.col_vbo = vbo.VBO(data=self.rgb, usage=GL.GL_DYNAMIC_DRAW, target=GL.GL_ARRAY_BUFFER)
        
    def drawObject(self):
        if not self.pos_vbo is None:
            # Update camera
            model, view, proj = glm.mat4(1), glm.mat4(1), glm.perspective(45, 640 / 480, 0.01, 100)        
            center, up, eye = glm.vec3(0,0,0), glm.vec3(0,-1,0), glm.vec3(0,0,-0.4 * (self.zoomLevel/10))
            view = glm.lookAt(eye, center, up)
            model = glm.rotate(model, self.xRot / 160.0, glm.vec3(1,0,0))
            model = glm.rotate(model, self.yRot / 160.0, glm.vec3(0,1,0))
            model = glm.rotate(model, self.zRot / 160.0, glm.vec3(0,0,1))
            mvp = proj * view * model
            GL.glUniformMatrix4fv(self.UNIFORM_LOCATIONS['mvp'], 1, False, glm.value_ptr(mvp))
    
            # Update data
            if not self.pos is None:
                self.pos_vbo.set_array(self.pos)
                
            if not self.rgb is None:
                self.col_vbo.set_array(self.rgb)
                # Color
                self.col_vbo.bind()
                GL.glEnableVertexAttribArray(1)
                GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            # Point size
            GL.glPointSize(4)
    
            # Position
            self.pos_vbo.bind()
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            # Draw
            GL.glDrawArrays(GL.GL_POINTS, 0, self.pos.shape[0])
    
            # Center debug
            if True:
                self.qglColor(QtGui.QColor(255,0,0))
                GL.glPointSize(20)
                GL.glBegin(GL.GL_POINTS)
                GL.glVertex3d(0,0,0)
                GL.glEnd()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    res = app.exec_()