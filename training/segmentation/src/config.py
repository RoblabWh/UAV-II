import numpy as np

# DATASET PARAMETERS
TRAIN_DIR = './new_Dataset/'
VAL_DIR = TRAIN_DIR
TRAIN_LIST = ['./list/train.txt'] * 3
VAL_LIST = ['./list/val.txt'] * 3
SHORTER_SIDE = [350] * 3
CROP_SIZE = [500] * 3
NORMALISE_PARAMS = [1./255, # SCALE
                    np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)), # MEAN
                    np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))] # STD
BATCH_SIZE = [4] * 3
NUM_WORKERS = 0
NUM_CLASSES = [21] * 3
LOW_SCALE = [0.5] * 3
HIGH_SCALE = [2.0] * 3
IGNORE_LABEL = 255

# ENCODER PARAMETERS
ENC = '152'
ENC_PRETRAINED = True  # pre-trained on ImageNet or randomly initialised

# GENERAL
EVALUATE = False
FREEZE_BN = [True] * 3
NUM_SEGM_EPOCHS = [100000] * 3
PRINT_EVERY = 500
RANDOM_SEED = 42
SNAPSHOT_DIR = './ckpt/'
CKPT_PATH = './ckpt/rf_voc_50_own.pth.tar'
VAL_EVERY = [10] * 3 # how often to record validation scores

# OPTIMISERS' PARAMETERS
LR_ENC = [5e-3, 2.5e-3, 1e-3]  # TO FREEZE, PUT 0
LR_DEC = [5e-2, 2.5e-2, 1e-2]
MOM_ENC = [0.85] * 3 # TO FREEZE, PUT 0
MOM_DEC = [0.85] * 3
WD_ENC = [1e-4] * 3 # TO FREEZE, PUT 0
WD_DEC = [1e-4] * 3
OPTIM_DEC = 'sgd'
