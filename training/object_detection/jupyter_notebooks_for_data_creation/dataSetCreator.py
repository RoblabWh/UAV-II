#### DataSetCreator ####

#Contains functions for the following activities:
##Rotation
##Blurring
##Coloring
##HomographyPerspective (needs a JSON file)
##HPDataCollector (for HomographyPerspective --> creates the JSON file with the necessary data)

import numpy as np
import cv2
import json
import io
import os

# angle maybe 15 and steps 0.1, scale should be 1.0
def rotateImage(src_image_path,  img_name, angle=15, steps=0.1, scale=1.0):
    img = cv2.imread(src_image_path+img_name)
    (height, width) = img.shape[:2]
    img_path = src_image_path
    img_height = height
    img_width = width
    img_center = (height/2, width/2)
    rotated_imgs = []
    # Peform rotation
    for i in np.arange(-angle,angle,steps):
        M = cv2.getRotationMatrix2D(img_center, i, scale)
        tmp_rot = cv2.warpAffine(img, M, (img_height, img_width))
        rotated_imgs.append(tmp_rot)
    return rotated_imgs

# possible gaussian kernels: (3,3), (5,5), (7,7), (9,9), (13,13), ...
def gaussianFilter(src_image_path, img_name, gaussianKernel):
    img = cv2.imread(src_image_path+img_name)
    img_path = src_image_path
    blur = cv2.GaussianBlur(img,(gaussianKernel,gaussianKernel),0)
    return blur

def gaussian_filter(img, gaussianKernel):
    blur = cv2.GaussianBlur(img,(gaussianKernel,gaussianKernel),0)
    return blur

# possible kernel sizes: 3, 5, 7, 9, 13, ...
def medianFilter(src_image_path, img_name, kernelSize):
    img = cv2.imread(src_image_path+img_name)
    median = cv2.medianBlur(self.img,kernelSize)
    return median

def median_filter(img, kernelSize):
    median = cv2.medianBlur(img, kernelSize)
    return median

def colorImageRed(src_image_path, img_name):
    img = cv2.imread(src_image_path+img_name)
    img[:,:,(0,1)] = 0
    return img

def color_image_red(img):
    img[:,:,(0,1)] = 0
    return img

def colorImageGreen(src_image_path, img_name):
    img = cv2.imread(src_image_path+img_name)
    g = img.copy()
    # set blue and red channels to 0
    g[:, :, [0, 2]] = 0
    return img

def color_image_green(img):
    g = img.copy()
    g[:,:, [0,2]] = 0
    return img

def colorImageGray(src_image_path, img_name):
    img = cv2.imread(src_image_path+img_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def color_image_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def invertImageColor(src_image_path, img_name):
    img = cv2.imread(src_image_path+img_name)
    return ~img

def invert_image_color(img):
    return ~img

def saveImages(output_name, img_path, img_list):
    for i in range(len(img_list)):
        cv2.imwrite(img_path+output_name+"_{}.jpg".format(i+1), img_list[i])

class HomoPerspective:
    def __init__(self, buildingData, src_image_path="", img_name=""):
        if src_image_path and img_name:
            self.img = cv2.imread(src_image_path+img_name)
            self.img_path = src_image_path
            self.height, self.width = self.img.shape[:2]
        self.buildingData = buildingData
        self.building = cv2.imread(buildingData["img_path"]+buildingData["img_name"])
        
    def passImage(self, img):
        self.img = img
        self.height, self.width = self.img.shape[:2]
    
    def combineImages(self):
        pts1=np.float32([[0,0],[self.width,0],[0,self.height],[self.width,self.height]])
        pts2=np.float32(self.buildingData["positions"])
        h, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        im1Reg = cv2.warpPerspective(self.img, h, (self.buildingData["width"], self.buildingData["height"]))
        mask2 = np.zeros(self.buildingData["shape"], dtype=np.uint8)
        roi_corners2 = np.int32(self.buildingData["positions2"])
        channel_count2 = self.buildingData["shape"][2]
        ignore_mask_color2 = (255,)*channel_count2
        cv2.fillConvexPoly(mask2, roi_corners2, ignore_mask_color2)
        mask2 = cv2.bitwise_not(mask2)
        masked_image2 = cv2.bitwise_and(self.building, mask2)
        final = cv2.bitwise_or(im1Reg, masked_image2)
        #cv2.imwrite('final.jpg', final)
        #cv2.imshow('FINAL', final)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return final
        
# Class for Collecting Data about backgrounds for Homography Persepective  

class HPDataCollector:
    def __init__(self, src_image_path, img_name):
        self.img = cv2.imread(src_image_path+img_name)
        self.img_path = src_image_path
        self.img_name = img_name
        self.positions = []
        self.positions2 = []
        self.count = 0
        self.buildings = {}
        self.data = dict()
        self.height, self.width = self.img.shape[:2]
        
        if os.path.exists('backgrounds.json'):
            print("File found and loaded into data:")
            with open('backgrounds.json') as f:
                data = json.load(f)
                print(data)
        else:
            print("No file found")

        
    # Mouse callback function
    def draw_circle(self,event,x,y,flags,param):
        # If event is Left Buttons Click then store the coordinate in the lists
        if event == cv2.EVENT_LBUTTONUP:
            cv2.circle(self.img,(x,y),2,(255,0,0),-1)
            self.positions.append([x,y])
            if(self.count!=3):
                self.positions2.append([x,y])
            elif(self.count==3):
                self.positions2.insert(2,[x,y])
            self.count+=1
            
    def makeNewData(self):
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',self.draw_circle)
        while(True):
            #imS = cv2.resize(building, (960, 720))     # Resize image
            cv2.imshow('image',self.img)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()
            
    def saveImageData(self):
        # check if file exists and if so, load it
        if os.path.exists(self.img_path+'backgrounds.json'):
            print("File found and loaded into data:")
            with open(self.img_path+'backgrounds.json') as f:
                self.data = json.load(f)
                print(self.data)
        else:
            print("No file found")

        # collect data of new image in an json object
        newBuilding = {}
        newBuilding["positions"] = self.positions
        newBuilding["positions2"] = self.positions2
        newBuilding["height"] = self.height
        newBuilding["width"] = self.width
        newBuilding["shape"] = self.img.shape
        newBuilding["img_path"] = self.img_path
        newBuilding["img_name"] = self.img_name

        # save data
        if not self.data:
            buildings = {}
            buildings[self.img_name] = newBuilding
            with io.open(os.path.join(self.img_path, 'backgrounds.json'), 'w') as db_file:
                db_file.write(json.dumps(buildings, sort_keys=True, indent=4))
        else:
            self.data[self.img_name] = newBuilding
            with io.open(os.path.join(self.img_path, 'backgrounds.json'), 'w') as db_file:
                db_file.write(json.dumps(self.data, sort_keys=True, indent=4))