import sys
import os
sys.path.append("/home/artur/Bibliotheken/tensorflow/models/research")
sys.path.append("/home/artur/Bibliotheken/tensorflow/models/research/object_detection")
sys.path.append("/home/artur/Bibliotheken/tensorflow/models/research/object_detection/utils")
import tensorflow as tf
from utils import ops as utils_ops
import numpy as np
import cv2

import six.moves.urllib as urllib
import tarfile

import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


import socket
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--host_ip', type=str, default=None)
parser.add_argument('--port', type=int, default=None)
args = parser.parse_args()

# This is needed since the notebook is stored in the object_detection folder.
#sys.path.append("..")


#if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
#  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

## Object detection Imports
from utils import label_map_util

from utils import visualization_utils as vis_util

def convBGRtoRGB(frame):
    b, g, r = cv2.split(frame)
    frame = cv2.merge((r, g, b))
    return frame

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_images(image, graph, tensor_dict, sess):
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def main():

        ## LOCAL OBJECT DETECTION
    MODEL_NAME = "warning_signs_graph"
    PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = os.path.join('data', 'object-detection.pbtxt')
    NUM_CLASSES = 3
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    # Client - Connection - Server
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2)
    server_address = (args.host_ip, args.port)
    
    
    #cap = cv2.VideoCapture(0)

    with detection_graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
                ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
            while True:
                #ret, image_np = cap.read()
                print("ObjectDetection: Try to recv...")
                try:
                    sent = sock.sendto("get".encode('utf-8'), server_address)
                    data, server = sock.recvfrom(65507)
                except:
                    print("ObjectDetection: Timeout...")
                    continue

                array = np.frombuffer(data, dtype=np.dtype('uint8'))
                image_np = cv2.imdecode(array, 1)
                image_np = convBGRtoRGB(image_np)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                output_dict = run_inference_images(image_np, detection_graph, tensor_dict, sess)
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    instance_masks=output_dict.get('detection_masks'),
                    use_normalized_coordinates=True,
                    line_thickness=8)
                cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break


main()
