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

NUM_CLASSES = 7

person_dataset = {"background" : True, "head": True, "upper body": True, "upper arm": True, "lowerarm": True, "upper legs": True, "lower legs": True}

class SemanticSegmentation:
    def __init__(self, cmap='cmap.npy', modelName='person50', host_ip='172.16.35.119', port=5555):
        print('constructor called')
        self.cmap = np.load(cmap)
        self.cuda = torch.cuda.is_available()
        self.model = self.getModel(modelName)
        self.host_ip = host_ip
        self.port = port
        self.labels = self.getLabels()

    def getLabels(self):
        labels = {}
        i = 0
        for key in person_dataset:
            if(self.cmap[i][0] == 0 and self.cmap[i][1] == 0 and self.cmap[i][2] == 0):
                pass
            else:
                new_label = {key: (self.cmap[i][0], self.cmap[i][1], self.cmap[i][2])}
                labels.update(new_label)
            i += 1
        return labels

    def filter(self, dataset):
        i = 0
        for val in dataset.items():
            if val[1] == 0:
                self.cmap[i] = [0,0,0]
            i += 1
        self.labels = self.getLabels()

    def __del__(self):
        print('destructor called')

    def getModel(self, modelName):
        if (modelName == 'person50'):
            models = {'rf_lw50_person': rf_lw50}
        elif (modelName == 'person101'):
            models = {'rf_lw101_person': rf_lw101}
        elif (modelName == 'person152'):
            models = {'rf_lw152_person': rf_lw152}
        model = dict()
        for key, fun in six.iteritems(models):
            net = fun(NUM_CLASSES, pretrained=True).eval()
            if self.cuda:
                net = net.cuda()
            model[key] = net
        return model

    def segmentation(self, frame):
        imageSize = frame.shape[:2][::-1]
        img_inp = torch.tensor(prepare_img(frame).transpose(2, 0, 1)[None]).float()
        if self.cuda:
            img_inp = img_inp.cuda()
        for mname, mnet in six.iteritems(self.model):
            segm = mnet(img_inp)[0].data.cpu().numpy().transpose(1, 2, 0)
            segm = cv2.resize(segm, imageSize, interpolation=cv2.INTER_CUBIC)
            # print(segm.argmax(axis=2).astype(np.uint8))
            # test = segm.argmax(axis=2).astype(np.uint8)
            segm = self.cmap[segm.argmax(axis=2).astype(np.uint8)]
        return segm

    def showImage(self, name, image, showAgenda=False):
        if (showAgenda):
            text_width = 10
            text_height = 20
            for key in self.labels:
                color = (int(self.labels[key][0]), int(self.labels[key][1]), int(self.labels[key][2]))
                cv2.putText(image, key, (text_width, text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                text_height += 20
        cv2.imshow(name, image)

    def listener(self, alpha=0.3):
        listen = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_adress = (self.host, self.port)
        beta = (1.0 - alpha)
        while (True):
            start_time = int(round(time.time() * 1000))
            sent = listen.sendto("get".encode('utf-8'), server_adress)
            data, server = listen.recvfrom(65507)
            # Error Msg weggelassen

            array = np.frombuffer(data, dtype=np.dtype('uint8'))
            image = cv2.imdecode(array, 1)
            segm = self.segmentation(image)
            overlay = cv2.addWeighted(image, alpha, segm, beta, 0.0)
            self.showImage("Semantische Segmentierung", overlay)
            end_time = int(round(time.time() * 1000))
            sys.stdout.write('\r Converted in {}s and {} ms'.format(int(round((end_time - start_time) / 1000)), (end_time - start_time) % 1000))

            sys.stdout.flush()
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                listen.sendto("quit".encode('utf-8'), server_adress)
                break
        cv2.destroyAllWindows()

    def showWebcam(self, video_port=0, alpha=0.3):
        #alpha = 0.3
        beta = (1.0 - alpha)
        cap = cv2.VideoCapture(video_port)

        while (True):
            start_time = int(round(time.time() * 1000))
            try:
                ret, frame = cap.read()
                segm = self.segmentation(frame)
                overlay = cv2.addWeighted(frame, alpha, segm, beta, 0.0)
                self.showImage("Raw Image", frame)
                self.showImage("Semantic Segmentation", segm)
                self.showImage("Overlay Image", overlay, True)
                end_time = int(round(time.time() * 1000))
                sys.stdout.write('\r Converted in {}s and {} ms'.format(int(round((end_time - start_time)/1000)),(end_time - start_time)%1000))
                sys.stdout.flush()

                if (cv2.waitKey(1) & 0xFF == ord('q')):
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


    def saveVideo(self):
        alpha = 0.3
        beta = (1.0 - alpha)
        path='videoOutput/'
        outputname='output'
        scale_percent = 50  # percent of original size


        cap = cv2.VideoCapture('15_39_16.avi')
        while(not cap.isOpened()):
            pass
        ret, frame = cap.read()
        #width = int(frame.shape[1])
        #height = int(frame.shape[0])

        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        #rame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fourccseg = cv2.VideoWriter_fourcc(*'XVID')

        over = cv2.VideoWriter('' + path + outputname + 'overlay.avi', fourcc, cap.get(cv2.CAP_PROP_FPS), dim)
        segmentation = cv2.VideoWriter('' + path + outputname + 'segm.avi', fourccseg, cap.get(cv2.CAP_PROP_FPS), dim)

        start_time = int(round(time.time() * 1000))
        #print('' + path + outputname + 'overlay/.avi')

        while(ret):
            """try:"""
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            segm = self.segmentation(frame)
            #print("hier")
            overlay = cv2.addWeighted(frame, alpha, segm, beta, 0.0)

            over.write(overlay)
            segmentation.write(segm)
            ret, frame = cap.read()

            #print("ret: {} frame: {}".format(ret, frame))

            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break
            """except:
                print("Unexpected error:", sys.exc_info()[0])
                break"""

        end_time = int(round(time.time() * 1000))
        delta_time = end_time - start_time
        ms = delta_time % 1000
        sec = int(round(delta_time / 1000))
        min = int(round(sec / 60))
        h = int(round(min / 60))

        sys.stdout.write('\r Converted in {}h, {}min, {}sec and {} ms'.format(h, min, sec, ms))
        sys.stdout.flush()

        cap.release()
        segmentation.release()
        over.release()

    def testImage(self):
        alpha = 0.3
        beta = (1.0 - alpha)
        image = cv2.imread('test.jpg', 1)
        if image is None:
            print("Unable to open " + 'test.jpg')
            exit(-1)

        scale_percent = 21  # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        segm = self.segmentation(image)
        overlay = cv2.addWeighted(image, alpha, segm, beta, 0.0)
        cv2.imwrite('overlayPP.jpg', overlay)
        cv2.imwrite('segmPP.jpg', segm)

def main():
    person = SemanticSegmentation()
    person.filter(person_dataset)
    person.testImage()
    #person.showWebcam()
    #voc.listener()


if __name__== "__main__":
    main()
