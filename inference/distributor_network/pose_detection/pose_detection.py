import tensorflow as tf
import cv2
import time
import argparse

import posenet
import time
import socket
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--host_ip', type=str, default=None)
parser.add_argument('--port', type=int, default=None)
args = parser.parse_args()

def convBGRto(frame):
    b, g, r = cv2.split(frame)
    frame = cv2.merge((r, g, b))
    return frame

def main():
    listen = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listen.settimeout(2)
    server_adress = (args.host_ip, args.port)

    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        #if args.file is not None:
        #    cap = cv2.VideoCapture(args.file)
        #else:
        #    cap = cv2.VideoCapture(args.cam_id)
        #cap.set(3, args.cam_width)
        #cap.set(4, args.cam_height)

        start = time.time()
        frame_count = 0
        
        while True:
            start_time = int(round(time.time() * 1000))
            print("PoseNet: Try to recv...")
            try:
                sent = listen.sendto("get".encode('utf-8'), server_adress)
                data, server = listen.recvfrom(65507)
            # Error Msg weggelassen
            except:
                print("PoseNet: Timeout...")
                continue

            array = np.frombuffer(data, dtype=np.dtype('uint8'))
            cap = cv2.imdecode(array, 1)
            cap = convBGRto(cap)
            input_image, display_image, output_scale = posenet.process(cap, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

            keypoint_coords *= output_scale

            # TODO this isn't particularly fast, use GL for drawing and display someday...
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            #cv2.namedWindow("posenet", cv2.WND_PROP_FULLSCREEN)
            #cv2.setWindowProperty("posenet",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            #cv2.imshow("posenet", overlay_image)
            cv2.imshow("PoseNet", overlay_image)
            end_time = int(round(time.time() * 1000))
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print('Average FPS: ', frame_count / (time.time() - start))
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
