import numpy as np
from utils_ import DepthNorm
from io import BytesIO
from PIL import Image
from zipfile import ZipFile
from tensorflow.keras.utils import Sequence
from augment import BasicPolicy
import glob
import os
import random
import cv2

def extract_zip(input_zip):
    input_zip=ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}

def nyu_resize(img, resolution=416, padding=6):
    from skimage.transform import resize
    return resize(img, (resolution, int(resolution*4/3)), preserve_range=True, mode='reflect', anti_aliasing=True )

def get_nyu_data(batch_size, nyu_data_zipfile='nyu_data.zip'):
    data = None

    #nyu2_train = list((row.split(',') for row in (data['data/nyu2_train.csv']).decode("utf-8").split('\n') if len(row) > 0))
    #nyu2_test = list((row.split(',') for row in (data['data/nyu2_test.csv']).decode("utf-8").split('\n') if len(row) > 0))

    dirName = '/media/gas/Samsung_T5/Datensätze/DatasetReduced_CroppedImages'

    nyu2_train = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        for file in filenames:
            if file.endswith('New.png'):
                nyu2_train.append((glob.glob(dirpath + '/*undist.ppm')[0], glob.glob(dirpath + '/*NoOverflow.pgm')[0], os.path.join(dirpath, file)))
    print(len(nyu2_train))


    print(len(nyu2_train))

    dirName = '/media/gas/Samsung_T5/Datensätze/KinectAufnahmenReduced_CroppedImages'

    #nyu2_train = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        for file in filenames:
            if file.endswith('PCLMask.png'):
                nyu2_train.append((glob.glob(dirpath + '/*undist.ppm')[0], glob.glob(dirpath + '/*projected.pgm')[0],
                                   os.path.join(dirpath, file)))
    print(len(nyu2_train))


    dirName = '/media/gas/Samsung_T5/Datensätze/DatasetReduced_CroppedEval'

    nyu2_test = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        for file in filenames:
            if file.endswith('New.png'):
                #print(os.path.join(dirpath + '*undist.ppm'))
                nyu2_test.append((glob.glob(dirpath + '/*undist.ppm')[0], glob.glob(dirpath + '/*NoOverflow.pgm')[0], os.path.join(dirpath, file)))
    print(len(nyu2_test))

    shape_rgb = (batch_size, 416,512, 3)
    shape_depth = (batch_size, 208, 256, 3)

    # Helpful for testing...
    if False:
        nyu2_train = nyu2_train[:10]
        nyu2_test = nyu2_test[:10]

    return data, nyu2_train, nyu2_test, shape_rgb, shape_depth

def get_nyu_train_test_data(batch_size):
    data, nyu2_train, nyu2_test, shape_rgb, shape_depth = get_nyu_data(batch_size)

    train_generator = NYU_BasicAugmentRGBSequence(data, nyu2_train, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)
    test_generator = NYU_BasicRGBSequence(data, nyu2_test, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)


    return train_generator, test_generator

class NYU_BasicAugmentRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size, shape_rgb, shape_depth, is_flip=False, is_addnoise=False, is_erase=False):
        self.data = data
        self.dataset = dataset
        self.policy = BasicPolicy( color_change_ratio=0.50, mirror_ratio=0.50, flip_ratio=0.0 if not is_flip else 0.2, 
                                    add_noise_peak=0 if not is_addnoise else 20, erase_ratio=-1.0 if not is_erase else 0.5)
        self.batch_size = batch_size
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.maxDepth = 65536.0

        from sklearn.utils import shuffle
        self.dataset = shuffle(self.dataset, random_state=0)

        self.N = len(self.dataset)
        # a1,a2,a3,...,an,a1,a2,a3,...,an,b1,b2,b3,...,bn,b1,b2,b3,...,bn,
        # 1 ... 50, 1 ... 50, - 451, ... 500, 451, ... 500 -- 1 ... 500 -- 1500 ... 2000 -- 1 ... 2000 --
        indices_a = [x for x in range(0, len(self.dataset))]

        n = 50
        indices_b = [indices_a[i * n:(i + 1) * n] for i in range((len(indices_a) + n - 1) // n )]

        indices_c = []
        for element in indices_b:
            indices_c.extend(element)
            indices_c.extend(element)

        n = 1000
        indices_b = [indices_c[i * n:(i + 1) * n] for i in range((len(indices_c) + n - 1) // n)]

        indices_c = []
        for element in indices_b:
            indices_c.extend(element)
            indices_c.extend(element)

        self.index = indices_c
        self.N = len(self.index)

        self.augmentation_params = {}
        for x in range(0, len(self.dataset)):
            self.augmentation_params[x] = (random.random(), random.random())

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx, is_apply_policy=True):
        batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth )

        # Augmentation of RGB images
        for i in range(batch_x.shape[0]):
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[self.index[index]]

            x = np.clip(np.asarray(Image.open( sample[0])).reshape(416,512,3)/255,0,1)
            y = np.clip(np.asarray(Image.open( sample[1])).reshape(208,256),0,self.maxDepth)

            planeSegments = np.clip(np.asarray(Image.open(sample[2])).reshape(208,256) / 255, 0, 1)
            x, y, planeSegments = self.policy(x, y, planeSegments, self.augmentation_params[self.index[index]][0], self.augmentation_params[self.index[index]][1])

            validMask = np.copy(y)
            validMask[validMask != 0.0] = 1.0
            y[y == 0.0] = self.maxDepth
            y = DepthNorm(y, maxDepth=self.maxDepth)

            x = cv2.cvtColor(np.asarray(x * 255, dtype=np.uint8), cv2.COLOR_RGB2HSV)/255

            batch_x[i] = x  # nyu_resize(x, 480)
            batch_y[i][..., 0] = y  # nyu_resize(y, 240)
            batch_y[i][..., 1] = validMask  # nyu_resize(y, 240)
            batch_y[i][..., 2] = planeSegments  # nyu_resize(y, 240)

            # DEBUG:
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()

        return batch_x, batch_y

class NYU_BasicRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size,shape_rgb, shape_depth):
        self.data = data
        self.dataset = dataset
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.maxDepth = 65536.0

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth )
        for i in range(self.batch_size):            
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]

            x = np.clip(np.asarray(Image.open(sample[0])).reshape(416,512,3) / 255, 0, 1)
            y = np.clip(np.asarray(Image.open(sample[1])).reshape(208,256), 0, self.maxDepth)
            planeSegments = np.clip(np.asarray(Image.open(sample[2])).reshape(208,256) / 255, 0, 1)


            validMask = np.copy(y)
            validMask[validMask != 0.0] = 1.0
            y[y == 0.0] = self.maxDepth

            y = DepthNorm(y, maxDepth=self.maxDepth)

            x = cv2.cvtColor(np.asarray(x * 255, dtype=np.uint8), cv2.COLOR_RGB2HSV)/255

            batch_x[i] = x  # nyu_resize(x, 480)
            batch_y[i][..., 0] = y  # nyu_resize(y, 240)
            batch_y[i][..., 1] = validMask  # nyu_resize(y, 240)
            batch_y[i][..., 2] = planeSegments  # nyu_resize(y, 240)

            # DEBUG:
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()

        return batch_x, batch_y

#================
# Unreal dataset
#================

#import cv2
from skimage.transform import resize

def get_unreal_data(batch_size, unreal_data_file='unreal_data.h5'):
    pass
    # shape_rgb = (batch_size, 480, 640, 3)
    # shape_depth = (batch_size, 240, 320, 1)
    #
    # # Open data file
    # import h5py
    # data = h5py.File(unreal_data_file, 'r')
    #
    # # Shuffle
    # from sklearn.utils import shuffle
    # keys = shuffle(list(data['x'].keys()), random_state=0)
    #
    # # Split some validation
    # unreal_train = keys[:len(keys)-100]
    # unreal_test = keys[len(keys)-100:]
    #
    # # Helpful for testing...
    # if False:
    #     unreal_train = unreal_train[:10]
    #     unreal_test = unreal_test[:10]
    #
    # return data, unreal_train, unreal_test, shape_rgb, shape_depth

def get_unreal_train_test_data(batch_size):
    data, unreal_train, unreal_test, shape_rgb, shape_depth = get_unreal_data(batch_size)
    
    train_generator = Unreal_BasicAugmentRGBSequence(data, unreal_train, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)
    test_generator = Unreal_BasicAugmentRGBSequence(data, unreal_test, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth, is_skip_policy=True)

    return train_generator, test_generator

class Unreal_BasicAugmentRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size, shape_rgb, shape_depth, is_flip=False, is_addnoise=False, is_erase=False, is_skip_policy=False):
        self.data = data
        self.dataset = dataset
        self.policy = BasicPolicy( color_change_ratio=0.50, mirror_ratio=0.50, flip_ratio=0.0 if not is_flip else 0.2, 
                                    add_noise_peak=0 if not is_addnoise else 20, erase_ratio=-1.0 if not is_erase else 0.5)
        self.batch_size = batch_size
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.maxDepth = 1000.0
        self.N = len(self.dataset)
        self.is_skip_policy = is_skip_policy

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx, is_apply_policy=True):
        batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth )
        
        # Useful for validation
        if self.is_skip_policy: is_apply_policy=False

        # Augmentation of RGB images
        for i in range(batch_x.shape[0]):
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]
            
            rgb_sample = cv2.imdecode(np.asarray(self.data['x/{}'.format(sample)]), 1)
            depth_sample = self.data['y/{}'.format(sample)] 
            depth_sample = resize(depth_sample, (self.shape_depth[1], self.shape_depth[2]), preserve_range=True, mode='reflect', anti_aliasing=True )
            
            x = np.clip(rgb_sample/255, 0, 1)
            y = np.clip(depth_sample, 10, self.maxDepth)
            y = DepthNorm(y, maxDepth=self.maxDepth)

            batch_x[i] = x
            batch_y[i] = y

            if is_apply_policy: batch_x[i], batch_y[i] = self.policy(batch_x[i], batch_y[i])
                
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i],self.maxDepth)/self.maxDepth,0,1), index, i)

        return batch_x, batch_y
