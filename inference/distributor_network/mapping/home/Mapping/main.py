import cv2
import numpy as np
import stat, os, argparse
import time
import threading
import multiprocessing
import math
from numba import jit
import ransac
import video_inference


@jit(nopython=True)
#  rotate with euler-rodrigues
#  rotates vector for angle around axis
def rotate_quat(vector, rotation):
    a, b, c, d = rotation
    aa = a * a
    bb = b * b
    cc = c * c
    dd = d * d
    ab = a * b
    ac = a * c
    ad = a * d
    bc = b * c
    bd = b * d
    cd = c * d
    rot_mat = np.array(((aa + bb - cc - dd, 2 * (bc - ad), 2 * (bd + ac)),
                        (2 * (bc + ad), aa + cc - bb - dd, 2 * (cd - ab)),
                        (2 * (bd - ac), 2 * (cd + ab), aa + dd - bb - cc)), dtype=np.float32)
    return np.dot(rot_mat, vector)


@jit(nopython=True)
def rotate_quat_orb(vector, rotation):
    x, y, z, w = rotation
    return rotate_quat(vector, (w, x, y, z))


def euler_to_quat(axis, angle):
    half_angle = angle / 2
    w = np.cos(half_angle)
    x, y, z = axis * np.sin(half_angle)
    return w, x, y, z


@jit(nopython=True)
#  rotate with euler-rodrigues
#  rotates vector for angle around axis
def rotate(vector, axis, angle):
    half_angle = angle/2
    a = np.cos(half_angle)
    b, c, d = axis * np.sin(half_angle)
    aa = a * a
    bb = b * b
    cc = c * c
    dd = d * d
    ab = a * b
    ac = a * c
    ad = a * d
    bc = b * c
    bd = b * d
    cd = c * d
    rot_mat = np.array(((aa + bb - cc - dd, 2 * (bc - ad), 2 * (bd + ac)),
                        (2 * (bc + ad), aa + cc - bb - dd, 2 * (cd - ab)),
                        (2 * (bd - ac), 2 * (cd + ab), aa + dd - bb - cc)), dtype=np.float32)
    return np.dot(rot_mat, vector)


@jit(nopython=True)
def rotate_x(vector, angle):
    a = math.cos(angle)
    b = math.sin(angle)
    rot_mat = np.array(((1, 0, 0),
                        (0, a, -b),
                        (0, b, a)), dtype=np.float32)
    return np.dot(rot_mat, vector)


@jit(nopython=True)
def rotate_y(vector, angle):
    a = math.cos(angle)
    b = math.sin(angle)
    rot_mat = np.array(((a, 0, b),
                        (0, 1, 0),
                        (-b, 0, a)), dtype=np.float32)
    return np.dot(rot_mat, vector)


@jit(nopython=True)
def rotate_z(vector, angle):
    a = math.cos(angle)
    b = math.sin(angle)
    rot_mat = np.array(((a, -b, 0),
                        (b, a, 0),
                        (0, 0, 1)), dtype=np.float32)
    return np.dot(rot_mat, vector)


def plane_intersection(LP0, LP1, n, d):
    u = LP1 - LP0
    w = LP0 - (n * d)
    D = np.dot(n, u)
    N = -np.dot(n, w)
    if abs(D) < 0.01:
        if N == 0:
            return None #'in plane' #line in plane
        else:
            return None #'is parallel' #line parallel to plane
    sI = N / D
    if sI < 0 or sI > 1:
        return None #'not in front'

    I = LP0 + sI * u
    return I


def relative_pixel(orig_pix, size, center='not used'):
    return (size[0] / 2  - orig_pix[0], orig_pix[1]  -  0.5 * size[1])


def segmentation_process(images, segments, max_keyframe, event):
    segmentation = video_inference.Segmentation()
    while True:
        event.wait()
        for image in images.keys():
            if image <= max_keyframe.value and image not in segments.keys():
                segments[image] = segmentation.get_segment(images[image])
        #print('max keyframe:', max_keyframe.value)
        event.clear()


def map(keyframes, points, images, segments, visualize_queue):
    x_axis = np.asarray([1.0, 0.0, 0.0])
    y_axis = np.asarray([0.0, 1.0, 0.0])
    #index of keyframe with lowest y pos
    #keyframe [keynumber, y, ?, ? , q? q? q? q?
    min_y_index = np.argmin(keyframes[:,2])#war mal 1
    print(min_y_index)
    angle = 0
    if keyframes[min_y_index, 2] != 0:
        angle = math.atan(keyframes[min_y_index, 2] / keyframes[min_y_index, 3])
        print(angle)

    #rotiert die keyframes
    rotated_keyframes = []
    for keyframe in keyframes:
        frame = keyframe.copy()
        frame[1:4] = rotate_x(frame[1:4], angle)
        rotated_keyframes.append(frame)
    rotated_keyframes = np.asarray(rotated_keyframes)

    #rotiere punkte und nehme die unterhalb der kameras
    rotated_and_sliced_points = []
    for point in points:
        pt = point.copy()
        pt = rotate_x(pt, angle)
        if pt[1] > 0:
            rotated_and_sliced_points.append(pt)
    rotated_and_sliced_points = np.asarray(rotated_and_sliced_points)


    points_under_path = len(rotated_and_sliced_points)
    print(points_under_path)
    if points_under_path > 20:
        n_max, d_max, max_points, min_outliers, inliers = ransac.ransac(rotated_and_sliced_points, 0.99, 0.8, 0.01)

        points_in_plane = []
        colors = []

        height = 720
        width = 960

        fx = 953.0
        fy = 932.5
        cx = 469.4
        cy = 334.0

        # fx = 920.0
        # fy = 920.0
        # cx = 480.0
        # cy = 360.0

        keyframe_vectors = []
        keyframe_colors = []
        for frame in rotated_keyframes:
            base_view = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
            view_direction = rotate(base_view, np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                                    angle)  # muss mit zum rotate keyframe
            view_direction = rotate_quat_orb(view_direction, frame[4:])
            oop = 0
            if frame[0] in segments.keys():
                img = images[frame[0]]
                seg = segments[frame[0]]
                for x in range(0, width, 10):
                    for y in range(0, height, 10):
                        if np.array_equal(seg[y, x], np.asarray([4, 200, 3])):  # prüft ob pixel xy auf dem boden liegt
                            yy = (y - cy) / fy
                            xx = (x - cx) / fx

                            pixel = np.asarray([xx, yy, 1.0], dtype=np.float32)
                            pixel = rotate_x(pixel, angle)
                            pixel = rotate_quat_orb(pixel, frame[4:])
                            point = plane_intersection(np.asarray(frame[1:4]),
                                                       np.asarray(frame[1:4]) + np.asarray(pixel),
                                                       n_max, d_max)
                            if point is not None:
                                # point = point - np.asarray(camera[0])
                                # point = Quaternion(axis=n_max, degrees=-90).rotate(point)
                                # point = point + np.asarray(camera[0])
                                points_in_plane.append(point)
                                colors.append(img[y, x])
                                # print(img[y, x])
                            else:
                                oop += 1
                                # print(point)
                        else:
                            oop += 1
                #print(len(points_in_plane), 'Points in Plane', oop, 'Points out of Plane')
            #keyframe_vectors.append(np.append(frame[1:4], [a + b for a, b in zip(frame[1:4], view_direction)]))
            keyframe_vectors.extend([frame[1:4], [a + b for a, b in zip(frame[1:4], view_direction)]])

        keyframe_lines = [[p, p + 1] for p in range(0, len(keyframe_vectors) - 1, 2)]
        if len(inliers) and len(points_in_plane) and len(colors) and len(keyframe_vectors) and len(keyframe_lines):
            points_in_plane = np.asarray(points_in_plane)

            visualize_queue.put([np.asarray(inliers), points_in_plane, np.asarray(colors), keyframe_vectors, keyframe_lines])


def input_thread(visualize_queue):
    count = 0
    while True:

        if not visualize_queue.empty():
            data = visualize_queue.get()
            inliers = data[0]
            points = data[1]
            colors = data[2]
            poses = data[3]
            lines = data[4]

            with open('./maps/map'+ str(count) + '.ply', 'w+') as filtered:
                filtered.write('ply\nformat ascii 1.0\nelement vertex ' + str(points.shape[0]) + '\nproperty double x\nproperty double y\nproperty double z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
                for point, color in zip(points, colors):
                    filtered.write(
                        str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + ' ' + str(color[0]) + ' ' + str(
                            color[1]) + ' ' + str(color[2]) + '\n')
            count += 1


            # inlier_cloud = o3d.PointCloud()
            # inlier_cloud.points = o3d.Vector3dVector(inliers)
            # point_cloud = o3d.PointCloud()
            # point_cloud.points = o3d.Vector3dVector(points)
            # point_cloud.colors = o3d.Vector3dVector(colors)
            # camera_poses_lines = o3d.geometry.LineSet()
            # camera_poses_lines.points = o3d.utility.Vector3dVector(poses)
            # camera_poses_lines.lines = o3d.utility.Vector2iVector(lines)
            # o3d.visualization.draw_geometries([inlier_cloud, point_cloud, camera_poses_lines])

        time.sleep(0.033)


def mapping(images, visualize_queue):
    print("Mapping thread started")
    manager = multiprocessing.Manager()
    segments = manager.dict()
    max_keyframe = manager.Value('i', 0)
    segmentation_event = multiprocessing.Event()
    keyframe_pipe = os.open("../ORB_SLAM2/Examples/Monocular/keyframe_fifo", os.O_RDONLY)
    point_pipe = os.open("../ORB_SLAM2/Examples/Monocular/point_fifo", os.O_RDONLY)
    print('opened point and keyframe pipes')
    seg_proc = multiprocessing.Process(target=segmentation_process, args=(images, segments, max_keyframe, segmentation_event))
    seg_proc.start()
    map_process = multiprocessing.Process(target=map)

    all_keyframe_numbers = set()
    print("Mapping thread running")
    while True:
        keyframes = os.read(keyframe_pipe, 1000000)
        keyframes = np.frombuffer(keyframes, dtype=np.float32)
        keyframes = keyframes.reshape((-1, 8))
        points = os.read(point_pipe, 1000000)
        points = np.frombuffer(points, dtype=np.float32)
        points = points.reshape((-1, 3))

        keyframe_numbers = list(keyframes[:, 0])
        all_keyframe_numbers.update(keyframe_numbers)
        keyframe_numbers.sort()

        #lösche bilder die keine keyframes(mehr) sind
        for key in images.keys():
            if key < keyframe_numbers[-1] and key not in all_keyframe_numbers:
                del images[key]

        #startet segmentierung wenn diese nicht läuft und ein nicht segmentiertes keyframe da ist
        if keyframe_numbers[-1] > max_keyframe.value and not segmentation_event.is_set():
            max_keyframe.value = keyframe_numbers[-1]
            segmentation_event.set()

        #should be event based and continously running, but with the numpy arrays keyframes and points, it's easier this way.
        if not map_process.is_alive() and keyframes.shape[0] > 3:
            map_process = multiprocessing.Process(target=map, args=(keyframes, points, images, segments, visualize_queue, ))
            map_process.start()


def main():

    parser = argparse.ArgumentParser(description='Mapping script')
    parser.add_argument('-i','--input', help='Video to be mapped', required=True)
    args = parser.parse_args()

    try:
        stat.S_ISFIFO(os.stat("../ORB_SLAM2/Examples/Monocular/image_fifo").st_mode)
    except FileNotFoundError:
        os.mkfifo("../ORB_SLAM2/Examples/Monocular/image_fifo")
        
    try:
        stat.S_ISFIFO(os.stat("../ORB_SLAM2/Examples/Monocular/keyframe_fifo").st_mode)
    except FileNotFoundError:
        os.mkfifo("../ORB_SLAM2/Examples/Monocular/keyframe_fifo")
        
        
    try:
        stat.S_ISFIFO(os.stat("../ORB_SLAM2/Examples/Monocular/point_fifo").st_mode)
    except FileNotFoundError:
        os.mkfifo("../ORB_SLAM2/Examples/Monocular/point_fifo")

        

    manager = multiprocessing.Manager()
    images = manager.dict()

    visualize_queue = manager.Queue()

    input_thread_ = multiprocessing.Process(target=input_thread, args=(visualize_queue, ))
    input_thread_.start()


    mapping_thread = multiprocessing.Process(target=mapping,
                                          args=(images, visualize_queue,))
    mapping_thread.start()

    cap = cv2.VideoCapture(args.input)

    # for i in range(0, 2400):
    #     disc = cap.grab()

    img_count = 0

    with open("../ORB_SLAM2/Examples/Monocular/image_fifo", "wb") as image_fifo:
        # with open("./test.txt", "wb") as image_fifo:
        while (True):
            before = time.time()
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_bytes = gray.tobytes()
            image_fifo.write(gray_bytes)
            images[img_count] = frame
            # Display the resulting frame
            # cv2.imshow('frame',cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
            after = time.time()
            img_count += 1

            if after - before < 0.033:
                time.sleep(0.033 - (after - before))
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # execute only if run as a script
    main()
