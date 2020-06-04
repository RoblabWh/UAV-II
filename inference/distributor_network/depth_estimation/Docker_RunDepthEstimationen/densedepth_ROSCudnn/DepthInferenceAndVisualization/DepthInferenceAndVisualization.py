import os
import sys
import glob
import time
import math
import argparse
import numpy as np
import rospy
import roslib
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

# Computer Vision
import cv2
from scipy import ndimage
from skimage.transform import resize

# Visualization
import matplotlib.pyplot as plt
plasma = plt.get_cmap('plasma')

# UI and OpenGL
from PySide2 import QtCore, QtGui, QtWidgets, QtOpenGL
from PySide2.QtCore import QThread
from OpenGL import GL, GLU
from OpenGL.arrays import vbo
from OpenGL.GL import shaders
import glm

from multiprocessing import Queue

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from tensorflow.keras.models import load_model
from layers import BilinearUpSampling2D

from tensorflow.python.compiler.tensorrt import trt_convert as trt
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
#last:
# /media/gas/Samsung_T5/NYU_DenseDepth/models/1578924287-n3468-e20-bs3-lr0.0001-densedepth_nyu/model.h5
#prev:
#/media/gas/Samsung_T5/NYU_DenseDepth/models/1578566569-n63-e20-bs3-lr0.0001-densedepth_nyu/model.h5
#40_40flur_model.h5
#

#weights.40-0.31.hdf5 DEKSTOP 1080 40 epochs from previous 40 epoch training based on trained nyu.h5
#weights.53-0.32.hdf5 DEKSTOP 1080 TI 40 Epochs from trained nyu.h5
#next smallDenseNet complete new training 38 epochs
parser.add_argument('--model', default='weights.53-0.32.hdf5', type=str, help='Trained Keras model file.') # nyu.h5
parser.add_argument('--viewing_mode', default='2', type=int, help='Different view schemes: 0 -rgb and depth image ; 1 - rgb and PCL ; 2 - rgb, depth image and PCL')
args = parser.parse_args()

# Image shapes
height_rgb, width_rgb = 480, 640
height_depth, width_depth = height_rgb // 2, width_rgb // 2
rgb_width = width_rgb
rgb_height = height_rgb

import tensorflow as tf

t1=time.time()
predictionRate = 0

# Function timing
ticTime = time.time()
def tic(): global ticTime; ticTime = time.time()
def toc(): print('{0} seconds.'.format(time.time() - ticTime))

# Conversion from Numpy to QImage and back
def np_to_qimage(a):
    im = a.copy()
    return QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888).copy()
    
def qimage_to_np(img):
    img = img.convertToFormat(QtGui.QImage.Format.Format_ARGB32)
    return np.array(img.constBits()).reshape(img.height(), img.width(), 4)

# Compute edge magnitudes
def edges(d):
    dx = ndimage.sobel(d, 0)  # horizontal derivative
    dy = ndimage.sobel(d, 1)  # vertical derivative
    return np.abs(dx) + np.abs(dy)

# Main window
class Window(QtWidgets.QWidget):
    updateVideoSignal = QtCore.Signal()

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.model = None
        self.capture = None
        self.glWidget = GLWidget()

        rospy.init_node('densedepth_node', anonymous=True)
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("/camera/rgb/image_rect_color", Image)
        self.depth_pub = rospy.Publisher("/camera/depth_registered/image_raw", Image)
        self.info_pub = rospy.Publisher("/camera/rgb/camera_info", CameraInfo)

        self.camera_info_msg = CameraInfo()
        width, height = 640, 480
        fx, fy = 935.02244588 * 2 / 3, 932.51652856 * 2 / 3
        cx, cy = 469.43202126 * 2 / 3, 333.97509921 * 2 / 3
        self.camera_info_msg.width = width
        self.camera_info_msg.height = height
        self.camera_info_msg.K = [fx, 0, cx,
                                  0, fy, cy,
                                  0, 0, 1]

        self.camera_info_msg.D = [0, 0, 0, 0]

        self.camera_info_msg.P = [fx, 0, cx, 0,
                                  0, fy, cy, 0,
                                  0, 0, 1, 0]

        mainLayout = QtWidgets.QVBoxLayout()

        # Input / output views
        viewsLayout = QtWidgets.QHBoxLayout()
        self.inputViewer = QtWidgets.QLabel("[Click to start]")
        self.inputViewer.setPixmap(QtGui.QPixmap(rgb_width,rgb_height))
        self.outputViewer = QtWidgets.QLabel("[Click to start]")
        self.outputViewer.setPixmap(QtGui.QPixmap(rgb_width,rgb_height))


        viewsLayout.addWidget(self.inputViewer)

        viewingMode = args.viewing_mode
        if viewingMode != 1:
            viewsLayout.addWidget(self.outputViewer)

        if viewingMode != 0:
            viewsLayout.addWidget(self.glWidget)

        mainLayout.addLayout(viewsLayout)

        # Load depth estimation model      
        toolsLayout = QtWidgets.QHBoxLayout()  

        self.button = QtWidgets.QPushButton("Load model...")
        self.button.clicked.connect(self.loadModel)
        toolsLayout.addWidget(self.button)

        self.button5 = QtWidgets.QPushButton("Load image...")
        self.button5.clicked.connect(self.loadImageFile)
        toolsLayout.addWidget(self.button5)
        self.button5.setEnabled(False)

        self.button3 = QtWidgets.QPushButton("Load Video...")
        self.button3.clicked.connect(self.loadVideoFile)
        toolsLayout.addWidget(self.button3)
        self.button3.setEnabled(False)

        self.button7 = QtWidgets.QPushButton("Skip Video 10s")
        self.button7.clicked.connect(self.skipFrames)
        toolsLayout.addWidget(self.button7)
        self.button7.setEnabled(False)

        self.button4 = QtWidgets.QPushButton("Stop Video")
        self.button4.clicked.connect(self.loadImage)
        toolsLayout.addWidget(self.button4)
        self.button4.setEnabled(False)

        mainLayout.addLayout(toolsLayout)

        self.setLayout(mainLayout)
        self.setWindowTitle(self.tr("DepthPrediction Viewer"))

        # Signals
        self.updateVideoSignal.connect(self.updateVideo)

        # Default example
        img = (self.glWidget.rgb * 255).astype('uint8')
        self.inputViewer.setPixmap(QtGui.QPixmap.fromImage(np_to_qimage(img)))
        coloredDepth = (plasma(resize(self.glWidget.depth[:,:,0], (rgb_height, rgb_width)))[:,:,:3] * 255).astype('uint8')

        self.outputViewer.setPixmap(QtGui.QPixmap.fromImage(np_to_qimage(coloredDepth)))

        self.currentImg = img
        self.nSkipSeconds = 10
        self.framerate = 30

        #self.imageQueue = Queue()
        #self.imgdepthQueue = Queue()
        #self.depthPredictor = DepthPredictor(self.imageQueue, self.imgdepthQueue)
        self.depthPredictor = DepthPredictor()
        '''
        self.depthPredictor.depthPredictedSignal.connect(self.updateCloud)
        self.depthPredictorThread = QtCore.QThread()
        self.depthPredictorThread.started.connect(self.depthPredictor.run)
        '''
        
    def loadModel(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(None, 'Select model', '', self.tr('Model files (*.h5 *.hdf5)'))[0]
        if filename:
            QtGui.QGuiApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            tic()
            if self.depthPredictor.loadModel(filename):
                #self.depthPredictor.moveToThread(self.depthPredictorThread)
                print('Model loaded.')
                toc()
                self.button5.setEnabled(True)
                #self.button4.setEnabled(True)
                #self.button7.setEnabled(True)
                self.button3.setEnabled(True)
                QtGui.QGuiApplication.restoreOverrideCursor()

                #self.depthPredictorThread.start()
            else:
                print('ERROR : Unable to load model')

    def loadCamera(self):        
        self.capture = cv2.VideoCapture(0)
        self.updateInput.emit()

    def loadVideoFile(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(None, 'Select Video', '', self.tr('Video files (*.avi)'))[0]
        print(filename)
        if filename:
            self.capture = cv2.VideoCapture(filename)
            if not self.button7.isEnabled():
                self.button7.setEnabled(True)
            #self.updateInput.emit()
            self.framerate = self.capture.get(cv2.CAP_PROP_FPS)
            self.updateVideoSignal.emit()

    def updateVideo(self):
        global predictionRate

        if self.capture is None: return

        ret, frame = self.capture.read()
        if not ret: self.loadImage()
        else:
            # Prepare image and show in UI
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (rgb_width, rgb_height), interpolation=cv2.INTER_AREA)
            self.currentImg = frame.copy()

            TopLeftCornerOfText = (15, 30)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            fontColor = (255, 255, 0)
            #cv2.putText(frame, '{:.2f} FPS'.format(predictionRate), TopLeftCornerOfText, font, fontScale, fontColor)



            image = np_to_qimage(frame)
            self.inputViewer.setPixmap(QtGui.QPixmap.fromImage(image))

            t1=time.time()
            depth = None

            depth = self.depthPredictor.doPrediction(self.currentImg)
            self.updateCloud(depth)
            t2=time.time()
            predictionRate = 1/(t2-t1)
            '''
            try:
                currentImg , depth = self.imgdepthQueue.get_nowait()
                #print(self.imgdepthQueue.empty())
                if depth is not None:
                    #print(type(currentImg), type(depth))
                    self.currentImg = currentImg
                    self.updateCloud(depth)
            except:
                pass
                #print(self.imgdepthQueue.empty())

            try:
                self.imgQueue.put_nowait(frame.copy())
            except:
                print(self.imgdepthQueue.empty())
            '''



            #self.depthPredictor.doPredictionSignal.emit(self.currentImg)
            if not self.button7.isEnabled():
                self.button7.setEnabled(True)
            if not self.button4.isEnabled():
                self.button4.setEnabled(True)
            QtCore.QTimer.singleShot(1000/self.framerate, self.updateVideo)

    def loadImage(self):
        self.capture = None
        self.button7.setEnabled(False)
        self.button4.setEnabled(True)

    def loadImageFile(self):
        self.capture = None
        filename = QtWidgets.QFileDialog.getOpenFileName(None, 'Select image', '', self.tr('Image files (*.png *.ppm)'))[0]
        if filename:
            img = QtGui.QImage(filename).scaled(rgb_width, rgb_height)
            xstart = 0
            if img.width() > rgb_width: xstart = (img.width() - rgb_width) // 2
            img = img.copy(xstart // 2, 0, xstart + rgb_width, rgb_height)
            self.inputViewer.setPixmap(QtGui.QPixmap.fromImage(img))
            currentImg = qimage_to_np(img)
            self.currentImg = currentImg
            depth = self.depthPredictor.doPrediction(self.currentImg)
            self.updateCloud(depth)

    def skipFrames(self):
        skippedFrames = 0
        while skippedFrames < self.nSkipSeconds * self.framerate:
            ret, frame = self.capture.read()
            skippedFrames+=1
            #print("ret: ", ret, " skipped Frames: ", skippedFrames)
        self.button7.setEnabled(False)
        self.button4.setEnabled(True)


    def updateCloud(self, depth):
        global t1
        global predictionRate

        # rgb8 = qimage_to_np(self.inputViewer.pixmap().toImage())
        if self.currentImg is not None:

            self.glWidget.rgb = resize((self.currentImg[:, :, :3] / 255)[:, :, :], (rgb_height, rgb_width), order=1)

            # send ros messages

            img_msg = self.bridge.cv2_to_imgmsg(resize(self.currentImg[:, :, :3], (rgb_height, rgb_width, 3), order=1, anti_aliasing=True,preserve_range=True).astype('uint8'), "rgb8")
            img_msg.header.frame_id = "camera_link"
            img_msg.header.stamp = rospy.Time.now()
            self.image_pub.publish(img_msg)
            depthToSend = depth.copy()
            sqDepth = np.squeeze(depthToSend).astype(np.float32)
            sqDepth[edges(sqDepth) > 0.3] = 0
            sqDepth[sqDepth > 4] = 0

            depth_msg = self.bridge.cv2_to_imgmsg(sqDepth, "32FC1")
            origDepth = depth[0,:,:,0]
            #print(origDepth.shape, np.max(origDepth), np.mean(origDepth), depth.shape,sqDepth.shape, np.max(sqDepth), np.mean(sqDepth))
            depth_msg.header.frame_id = "camera_link"
            depth_msg.header.stamp = rospy.Time.now()
            self.depth_pub.publish(depth_msg)
            self.camera_info_msg.header.stamp = rospy.Time.now()
            self.info_pub.publish(self.camera_info_msg)

            coloredDepth = np.copy(depth)
            coloredDepth[coloredDepth > 6] = 6
            coloredDepth = (plasma(0.15 * resize(coloredDepth[0, :, :, 0], (rgb_height, rgb_width)))[:, :, :3] * 255).astype('uint8')
            self.outputViewer.setPixmap(QtGui.QPixmap.fromImage(np_to_qimage(coloredDepth)))
            self.glWidget.depth = depth[0, :, :, 0]

        else:
            self.glWidget.depth = 0.5 + np.zeros((rgb_height, rgb_width, 1))

        self.glWidget.updateRGBD()
        self.glWidget.updateGL()

class DepthPredictor(QtCore.QObject):
    #doPredictionSignal = QtCore.Signal(np.ndarray)
    #depthPredictedSignal = QtCore.Signal(np.ndarray)

    #def __init__(self, imageQueue, imgdepthQueue, parent=None):
    def __init__(self, parent=None):
        QtCore.QObject.__init__(self,parent)
        #self.doPredictionSignal.connect(self.doPrediction)
        self.model = None
        self.graph = None

        #self.imageQueue = imageQueue
        #self.imgdepthQueue = imgdepthQueue


    def loadModel(self, modelFilename):
        modelLoaded = False
        if os.path.exists(modelFilename):
            if modelFilename.endswith('pb'):
                input_saved_model_dir = os.path.split(os.path.realpath(__file__))[0]
                converter = trt.TrtGraphConverterV2(input_saved_model_dir=input_saved_model_dir)
                converter.convert()
                output_saved_model_dir = input_saved_model_dir
                converter.save(output_saved_model_dir)
            if modelFilename.endswith('h5') or modelFilename.endswith('hdf5'):
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

                self.graph = tf.compat.v1.get_default_graph()

                with self.graph.as_default():
                    # Custom object needed for inference and training
                    custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

                    # Load model into GPU / CPU
                    self.model = load_model(modelFilename, custom_objects=custom_objects, compile=False)
                    modelLoaded = True
                    #self.model.save(os.path.split(os.path.realpath(__file__))[0])

        return modelLoaded

    def run(self):


        while True:

            img = None
            depth = None
            try:
                while True:
                    imgTmp = self.imgQueue.get_nowait()
                    print(type(imgTmp))
                    img = imgTmp

            except:
                if img is not None:

                    depth = self.doPrediction(img)

            try:
                if not img is None and not depth is None:
                    self.imgdepthQueue.put_nowait([img,depth])
                    img=None
                    depth=None
            except:
                pass

            time.sleep(0.01)
        #QtCore.QTimer.singleShot(10, self.run)

    def doPrediction(self, img):
        depth = None

        if self.model:
            with self.graph.as_default():
                rgb8 = resize((img[:, :, :3] / 255)[:, :, :], (rgb_height, rgb_width), order=1)
                depth = 10 / self.model.predict(np.expand_dims(rgb8, axis=0))

        return depth


class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        QtOpenGL.QGLWidget.__init__(self, parent)

        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.object = 0
        #self.xRot = 5040
        #self.yRot = 40
        self.xRot = 0
        self.yRot = 0
        self.zRot = 0
        self.zoomLevel = 9

        self.lastPos = QtCore.QPoint()

        self.trolltechGreen = QtGui.QColor.fromCmykF(0.40, 0.0, 1.0, 0.0)
        self.trolltechPurple = QtGui.QColor.fromCmykF(0.39, 0.39, 0.0, 0.0)



        # Precompute for world coordinates 
        self.xx, self.yy = self.worldCoords(width=rgb_width//2, height=rgb_height//2)

        # Load test frame from disk
        self.rgb = np.load('demo_rgb.npy')
        self.depth = resize(np.load('demo_depth.npy'), (240,320))
        self.col_vbo = None
        self.pos_vbo = None
        self.posDelta = glm.vec3(0,0,0)
        self.updateRGBD()

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
            self.emit(QtCore.SIGNAL("xRotationChanged(int)"), angle)
            self.updateGL()

    def setYRotation(self, angle):
        if angle != self.yRot:
            self.yRot = angle
            self.emit(QtCore.SIGNAL("yRotationChanged(int)"), angle)
            self.updateGL()

    def setZRotation(self, angle):
        if angle != self.zRot:
            self.zRot = angle
            self.emit(QtCore.SIGNAL("zRotationChanged(int)"), angle)
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
        numDegrees = event.delta() / 8
        numSteps = numDegrees / 15
        self.zoomLevel = self.zoomLevel + numSteps
        event.accept()
        self.updateGL()

    def rotation_matrix(self, axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = np.asarray(axis)
        axis = axis / math.sqrt(np.dot(axis, axis))
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    def keyPressEvent(self, event):
        '''
        Strange behaviour in Widget
        '''

        '''
        view = np.identity(3)
        rot = np.dot(view, self.rotation_matrix(np.array((1,0,0)), self.xRot / 160.0))
        rot = np.dot(rot, self.rotation_matrix(np.array((0,1,0)), self.yRot / 160.0))
        rot = np.dot(rot, self.rotation_matrix(np.array((0,0,1)), self.zRot / 160.0))

        posDelta = None
        if event.key() == QtCore.Qt.Key_Up:
            #posDelta = glm.vec3(0,0, 0.01/self.zoomLevel)
            posDelta = np.array((0, 0, 0.01 / self.zoomLevel))
        elif event.key() == QtCore.Qt.Key_Down:
            posDelta = np.array((0,0, -0.01/self.zoomLevel))
        elif event.key() == QtCore.Qt.Key_Left:
            posDelta = np.array((-0.01/self.zoomLevel,0,0))
        elif event.key() == QtCore.Qt.Key_Right:
            posDelta = np.array((0.01/self.zoomLevel,0,0))

        posDelta = np.dot(rot, posDelta)
        #print(posDelta)
        self.posDelta = self.posDelta + glm.vec3(posDelta[0], posDelta[1], posDelta[2])
        self.pos = self.pos + self.posDelta
        self.updateGL()
        '''

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

    def worldCoords(self, width, height):
        '''
        hfov_degrees, vfov_degrees = 57, 43
        hFov = math.radians(hfov_degrees)
        vFov = math.radians(vfov_degrees)
        cx, cy = width/2, height/2
        fx = width/(2*math.tan(hFov/2))
        fy = height/(2*math.tan(vFov/2))
        '''
        # 960 x 720
        # 640 # 480
        fx = 935.02244588 / 1.5
        fy = 932.51652856 / 1.5
        fx /= 2
        fy /= 2
        cx = (469.43202126 / 1.5) / 4
        cy = (333.97509921 / 3) / 4
        # cx=480
        # cy=360
        xx, yy = np.tile(range(width), height), np.repeat(range(height), width)
        xx = (xx - cx) / fx
        yy = (yy - cy) / fy
        return xx, yy

    def posFromDepth(self, depth):
        length = depth.shape[0] * depth.shape[1]

        depth[edges(depth) > 0.8] = 0
        depth[depth > 4] = 0
        z = depth.reshape(length)

        st = np.dstack((self.xx * z, self.yy * z, z)).reshape((length, 3))
        # st=np.array([p if p[2]<4 else np.zeros(3,dtype=type(p[0])) for p in st])
        return st

    def createPointCloudVBOfromRGBD(self):
        # Create position and color VBOs
        self.pos_vbo = vbo.VBO(data=self.pos, usage=GL.GL_DYNAMIC_DRAW, target=GL.GL_ARRAY_BUFFER)
        self.col_vbo = vbo.VBO(data=self.col, usage=GL.GL_DYNAMIC_DRAW, target=GL.GL_ARRAY_BUFFER)

    def updateRGBD(self):
        # RGBD dimensions
        width, height = self.depth.shape[1], self.depth.shape[0]

        # Reshape
        points = self.posFromDepth(self.depth.copy())
        colors = resize(self.rgb, (height, width)).reshape((height * width, 3))


        # Flatten and convert to float32
        self.pos = points.astype('float32')
        self.col = colors.reshape(height * width, 3).astype('float32')

        # Move center of scene
        self.pos = self.pos + glm.vec3(-0.3,-0.2, -1)

        # Create VBOs
        if not self.col_vbo:
            self.createPointCloudVBOfromRGBD()

    def drawObject(self):
        # Update camera
        model, view, proj = glm.mat4(1), glm.mat4(1), glm.perspective(45, self.width() / self.height(), 0.01, 100)
        center, up, eye = glm.vec3(0,0,-0.2), glm.vec3(0,-1,0), glm.vec3(0,0,-0.4 * (self.zoomLevel/10))
        #center, up, eye = glm.vec3(0, 0, 0) , glm.vec3(0, -1, 0), glm.vec3(0, 0, 0) + self.posDelta
        view = glm.lookAt(eye, center, up)
        model = glm.rotate(model, self.xRot / 160.0, glm.vec3(1,0,0))
        model = glm.rotate(model, self.yRot / 160.0, glm.vec3(0,1,0))
        model = glm.rotate(model, self.zRot / 160.0, glm.vec3(0,0,1))

        mvp = proj * view * model
        GL.glUniformMatrix4fv(self.UNIFORM_LOCATIONS['mvp'], 1, False, glm.value_ptr(mvp))
        #self.pos = self.pos + glm.vec3(0,-0.06,-0.3)
        # Update data
        self.pos_vbo.set_array(self.pos)
        self.col_vbo.set_array(self.col)

        # Point size
        GL.glPointSize(4)

        # Position
        self.pos_vbo.bind()
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        # Color
        self.col_vbo.bind()
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        # Draw
        GL.glDrawArrays(GL.GL_POINTS, 0, self.pos.shape[0])

        # Center debug
        if False:
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
