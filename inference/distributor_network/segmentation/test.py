import six
import sys
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

#sys.path.append('../../')

from models.mobilenet import mbv2
from models.resnet import rf_lw50, rf_lw101, rf_lw152
from utils.helpers import prepare_img
from PIL import Image

def calcPlot(models, imgs, has_cuda, cmap):
    n_cols = len(models) + 1 # 1 - for image, 1 - for GT
    n_rows = len(imgs)

    plt.figure(figsize=(16, 12))
    idx = 1

    alpha = 0.3
    beta = (1.0 - alpha)

    with torch.no_grad():
        for img_path in imgs:
            img = np.array(Image.open(img_path))
            #msk = cmap[np.array(Image.open(img_path.replace('jpg', 'png')))]
            orig_size = img.shape[:2][::-1]

            img_inp = torch.tensor(prepare_img(img).transpose(2, 0, 1)[None]).float()
            if has_cuda:
                img_inp = img_inp.cuda()

            plt.subplot(n_rows, n_cols, idx)
            plt.imshow(img)
            plt.title('img')
            plt.axis('off')
            idx += 1

            #plt.subplot(n_rows, n_cols, idx)
            #plt.imshow(msk)
            #plt.title('gt')
            #plt.axis('off')
            #idx += 1

            for mname, mnet in six.iteritems(models):
                segm = mnet(img_inp)[0].data.cpu().numpy().transpose(1, 2, 0)
                segm = cv2.resize(segm, orig_size, interpolation=cv2.INTER_CUBIC)
                segm = cmap[segm.argmax(axis=2).astype(np.uint8)]

                overlay = cv2.addWeighted(img, alpha, segm, beta, 0.0)
                cv2.imwrite('Result/' + str(idx) + '.jpg', overlay)


                plt.subplot(n_rows, n_cols, idx)
                plt.imshow(segm)
                plt.title(mname)
                plt.axis('off')
                idx += 1

def loadImg(path):
    img_dir = path
    imgs = glob.glob('{}*.jpg'.format(img_dir))
    return imgs

def loadCmap():
    cmap = np.load('cmap.npy')
    cuda = torch.cuda.is_available()
    return cmap, cuda

def init(mode, cuda):
    if (mode == 1):
        model= { 'rf_lw50_voc'   : rf_lw50 }
    elif (mode == 2):
        model= { 'rf_lw101_voc'  : rf_lw101 }
    elif (mode == 3):
        model = { 'rf_lw152_voc'  : rf_lw152 }
    elif (mode == 4):
        model = { 'rf_lwmbv2_voc': mbv2 }
    models = dict()
    for key,fun in six.iteritems(model):
        net = fun(21, pretrained=True).eval()
        if cuda:
            net = net.cuda()
        models[key] = net
    return models

def main():
    path = 'VOC/'
    mode = 4
    start_time = int(round(time.time() * 1000))
    cmap, cuda = loadCmap()
    imgs = loadImg(path)
    models = init(1, cuda)
    calcPlot(models, imgs, cuda, cmap)
    end_time = int(round(time.time() * 1000))
    print("{} Dataset convertet in {}s and {} ms".format(models, int(round((end_time - start_time)/1000)),(end_time - start_time)%1000))



if __name__== "__main__":
    main()
