import six
import sys
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import socket

#sys.path.append('../../')

from models.mobilenet import mbv2
from models.resnet import rf_lw50, rf_lw101, rf_lw152
from utils.helpers import prepare_img
from PIL import Image

def listener(cuda, cmap, model, server_adress, listen):
    alpha = 0.3
    beta = (1.0 - alpha)
    while(True):
        start_time = int(round(time.time() * 1000))
        sent = listen.sendto("get".encode('utf-8'), server_adress)
        data, server = listen.recvfrom(65507)
        #Error Msg weggelassen

        array = np.frombuffer(data, dtype=np.dtype('uint8'))
        image = cv2.imdecode(array, 1)
        imageRGB = image
        imageSize = image.shape[:2][::-1]
        img_inp = torch.tensor(prepare_img(image).transpose(2, 0, 1)[None]).float()
        if cuda:
            img_inp = img_inp.cuda()
        for mname, mnet in six.iteritems(model):
                segm = mnet(img_inp)[0].data.cpu().numpy().transpose(1, 2, 0)
                segm = cv2.resize(segm, imageSize, interpolation=cv2.INTER_CUBIC)
                segm = cmap[segm.argmax(axis=2).astype(np.uint8)]
        #Image,Text,position, font,Größe,color, Fett, ?
        #cv2.putText(imageRGB,text,(50,50), font, 1,(1,255,255),5,cv2.LINE_AA)
        # Display the resulting frame
        overlay = cv2.addWeighted(imageRGB, alpha, segm, beta, 0.0)
        end_time = int(round(time.time() * 1000))
        sys.stdout.write('\r Converted in {}s and {} ms'.format(int(round((end_time - start_time)/1000)),(end_time - start_time)%1000))

        #print("\r Converted in in {}s and {} ms".format(int(round((end_time - start_time)/1000)),(end_time - start_time)%1000))
        sys.stdout.flush()
        cv2.imshow("Semantische Segmentierung", overlay)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            listen.sendto("quit".encode('utf-8'), server_adress)
            break
    #cap.release()
    cv2.destroyAllWindows()

def webcam(cuda, cmap, model):
    alpha = 0.3
    beta = (1.0 - alpha)
    i = 0
    cap = cv2.VideoCapture(0)

    while(True):
        #start_time = int(round(time.time() * 1000))
        try:

            ret, frame = cap.read()
            imageRGB = frame
            imageSize = frame.shape[:2][::-1]
            img_inp = torch.tensor(prepare_img(frame).transpose(2, 0, 1)[None]).float()
            if cuda:
                img_inp = img_inp.cuda()
            for mname, mnet in six.iteritems(model):
                segm = mnet(img_inp)[0].data.cpu().numpy().transpose(1, 2, 0)
                segm = cv2.resize(segm, imageSize, interpolation=cv2.INTER_CUBIC)
                #print(segm.argmax(axis=2).astype(np.uint8))
                test = segm.argmax(axis=2).astype(np.uint8)
                segm = cmap[segm.argmax(axis=2).astype(np.uint8)]

            #w,h =test.shape[:2]
            #print(test)
            #print(test.shape)
            x1, x2, y1, y2 = boundingBox(15, test)

            #break
                #print()
            #Image,Text,position, font,Größe,color, Fett, ?
            #cv2.putText(imageRGB,text,(50,50), font, 1,(1,255,255),5,cv2.LINE_AA)
            # Display the resulting frame
            overlay = cv2.addWeighted(imageRGB, alpha, segm, beta, 0.0)

            cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,0,255),1)
            #end_time = int(round(time.time() * 1000))
            #sys.stdout.write('\r Converted in {}s and {} ms'.format(int(round((end_time - start_time)/1000)),(end_time - start_time)%1000))

            #print("\r Converted in in {}s and {} ms".format(int(round((end_time - start_time)/1000)),(end_time - start_time)%1000))
            #sys.stdout.flush()
            cv2.imshow("Semantische Segmentierung" ,overlay)
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                #print(cmap)
                break

        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise
            print("Exception throw")
            cv2.destroyAllWindows()
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def saveImg(imgs, cuda, cmap, model):
    n_cols = len(model) + 1 # 1 - for image, 1 - for GT
    n_rows = len(imgs)

    plt.figure(figsize=(16, 12))
    idx = 0

    alpha = 0.3
    beta = (1.0 - alpha)

    with torch.no_grad():
        for img_path in imgs:
            img = np.array(Image.open(img_path))
            #msk = cmap[np.array(Image.open(img_path.replace('jpg', 'png')))]
            orig_size = img.shape[:2][::-1]
            img_inp = torch.tensor(prepare_img(img).transpose(2, 0, 1)[None]).float()
            if cuda:
                img_inp = img_inp.cuda()
            for mname, mnet in six.iteritems(model):
                segm = mnet(img_inp)[0].data.cpu().numpy().transpose(1, 2, 0)
                segm = cv2.resize(segm, orig_size, interpolation=cv2.INTER_CUBIC)
                segm = cmap[segm.argmax(axis=2).astype(np.uint8)]

                overlay = cv2.addWeighted(img, alpha, segm, beta, 0.0)
                cv2.imwrite('Result/' + str(idx) + '.jpg', overlay)
            idx += 1

def boundingBox(value, img):
    width, height = img.shape[:2]
    #print(img)
    x1 = width
    y1 = 0
    x2 = 0
    y2 = height
    for y in range(0, height):
        for x in range(0, width):
            if(img[x,y] == 15):
                #print ("gefunden")
                if (x < x1):
                    x1 = x
                if(y > y1):
                    y1 = y
                if(x > x2):
                    x2 = x
                if(y < y2):
                    y2 = y
    return x1, x2, y1, y2


def loadImg(path):
    img_dir = path
    imgs = glob.glob('{}*.jpg'.format(img_dir))
    return imgs

def initCmap():
    cmap = np.load('cmap.npy')
    return cmap

def initCuda():
    cuda = torch.cuda.is_available()

    return cuda

def initModel(cuda, modelType):
    if (modelType == 1):
        models= { 'rf_lw50_voc'   : rf_lw50 }
    elif (modelType == 2):
        models= { 'rf_lw101_voc'  : rf_lw101 }
    elif (modelType == 3):
        models = { 'rf_lw152_voc'  : rf_lw152 }
    elif (modelType == 4):
        models = { 'rf_lwmbv2_voc': mbv2 }
    model = dict()
    for key,fun in six.iteritems(models):
        net = fun(21, pretrained=True).eval()
        if cuda:
            net = net.cuda()
        model[key] = net
    return model

def initSocket(host, port):
    listen = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_adress = (host, port)
    return listen, server_adress

def main():
    modus = 2
    modelType = 1
    path = 'VOC/'

    host = '172.16.35.119'
    port = 5555

    torch.cuda.empty_cache()

    cuda = initCuda()
    cmap = initCmap()
    model = initModel(cuda, modelType)

    if(modus == 1):
        imgs = loadImg(path)
        saveImg(imgs, cuda, cmap, model)
    elif(modus == 2):
        webcam(cuda, cmap, model)
    elif(modus == 3):
        listen, server_adress = initSocket(host, port)
        listener(cuda, cmap, model, server_adress, listen)


if __name__== "__main__":
    main()
