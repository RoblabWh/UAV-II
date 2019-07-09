
import numpy as np
import cv2
import sys
import traceback
import tellopy
import av
import time
import datetime
import os

CountImagesWithTrackedCorners = 20
chessboardSize = (8,6) #count inner corners, horizontal first
squareWidthHeight = 24 #mm

def grabCornersImage(cap, chessboardSize):

    cornersFound = False

    corners = None
    img = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) #subpixel interpolation

    ret, frame = cap.read()
    if ret == True:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, cornersPre = cv2.findChessboardCorners(gray, chessboardSize, None)
        if ret:
            # Draw and display the corners

            corners = cv2.cornerSubPix(gray, cornersPre, (11, 11), (-1, -1), criteria)

            imgCorners = cv2.drawChessboardCorners(gray, chessboardSize, corners, ret)
            #cv2.imshow('corners Frame', imgCorners)
            #cv2.waitKey(500)

            img = gray
            cornersFound = True


    return cornersFound, corners, img


def calibrateCamera(cap):
    objPointsChessLocal = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objPointsChessLocal[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

    objectsPointsChessLocal = []
    imagePoints = []

    # grab images for calibration
    numberFramesFound = 0
    while (numberFramesFound <= CountImagesWithTrackedCorners):
        ret, corners, img = grabCornersImage(cap, chessboardSize)
        if ret:
            imagePoints.append(corners)
            objectsPointsChessLocal.append(objPointsChessLocal)

            numberFramesFound += 1

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, chessboardSize, corners, ret)
            #cv2.imshow('img', img)
            #cv2.waitKey(500)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectsPointsChessLocal , imagePoints, img.shape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs

def calibrateCameraImages(grabbedImgs):

    objPointsChessLocal = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objPointsChessLocal[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

    objectsPointsChessLocal = []
    imagePoints = []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # subpixel interpolation
    imgShape = None

    for i, img in enumerate(grabbedImgs):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if i == 0:
            imgShape = gray.shape[::-1]

        ret, cornersPre = cv2.findChessboardCorners(gray, chessboardSize, None)
        if ret:
            # Draw and display the corners

            corners = cv2.cornerSubPix(gray, cornersPre, (11, 11), (-1, -1), criteria)

            imgCorners = cv2.drawChessboardCorners(gray, chessboardSize, corners, ret)
            cv2.imshow('corners Frame', imgCorners)
            cv2.waitKey(500)

            imagePoints.append(corners)
            objectsPointsChessLocal.append(objPointsChessLocal)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectsPointsChessLocal, imagePoints, imgShape, None, None)

    return ret, mtx, dist, rvecs, tvecs

def grabImagesSystem(camIndex=0):
    grabbedImgs = []
    aborted = False

    # Create a VideoCapture and read from std camera
    cap = cv2.VideoCapture(camIndex)
    if (cap.isOpened() == False):
        raise Exception("Error opening video stream or file")

    print("type 'c' for an image you want to use for calibration")
    while not aborted and cap.isOpened() and len(grabbedImgs) <= CountImagesWithTrackedCorners:
        ret, img = cap.read()
        if ret:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Frame', gray)
            k = cv2.waitKey(50)
            print(k)
            if k == ord('c'):
                grabbedImgs.append(img)

            elif k == 27:
                aborted = True

    # When everything done, release the video capture object
    cap.release()

    return grabbedImgs, aborted

def grabImagesTello():
    grabbedImgs = []
    aborted = False
    sufficientImgCount = False

    drone = tellopy.Tello()

    try:
        drone.connect()
        drone.wait_for_connection(60.0)

        retry = 3
        container = None
        while container is None and 0 < retry:
            retry -= 1
            try:
                container = av.open(drone.get_video_stream())
            except av.AVError as ave:
                print(ave)
                print('retry...')

        # skip first 300 frames
        frame_skip = 300
        while not aborted and not sufficientImgCount:
            for frame in container.decode(video=0):
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
                start_time = time.time()
                image = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)
                cv2.imshow('Original', image)

                if frame.time_base < 1.0 / 60:
                    time_base = 1.0 / 60
                else:
                    time_base = frame.time_base
                frame_skip = int((time.time() - start_time) / time_base)

                k = cv2.waitKey(1)
                if k == 27:
                    aborted = True
                    break
                elif k == 13 or k == 32:
                    grabbedImgs.append(image)
                    if len(grabbedImgs) == CountImagesWithTrackedCorners:
                        sufficientImgCount = True
                        break


    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
        aborted = True
        grabbedImgs = []
    finally:
        drone.quit()
        cv2.destroyAllWindows()

    return grabbedImgs, aborted

def main():

    aborted = False
    grabbedImgs = []


    grabbedImgs, aborted = grabImagesTello()
    #grabbedImgs, aborted = grabImagesSystem(0)


    if not aborted:
        ret, mtx, dist, rvecs, tvecs = calibrateCameraImages(grabbedImgs)

        print(ret)
        print()
        print(mtx)
        print()
        print(dist)
        print(np.array(rvecs).shape)
        print()
        print(np.array(tvecs).shape)

        nowTime = datetime.datetime.now()
        nowTimeText = 'y{}_M{}_{}d_{}h_{}m_{}s'.format(nowTime.year, nowTime.month, nowTime.day, nowTime.hour, nowTime.minute, nowTime.second)
        dataDir = r"./calibrationImages_{}".format(nowTimeText)
        os.mkdir(dataDir)

        np.savez(os.path.join(dataDir, 'calib.npz'), mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
        f = open(os.path.join(dataDir, 'calib.dat'), 'a')
        f.write('{}\n'.format(ret))
        f.write('{}\n'.format(mtx))
        f.write('{}\n'.format(dist))
        f.write('{}\n'.format(rvecs))
        f.write('{}\n\n'.format(tvecs))

        #np.savetxt(f, np.ndarray(shape=(1,1), buffer=np.array((ret)) ))
        #np.savetxt(f, mtx)
        #np.savetxt(f, dist)
        #np.savetxt(f, rvecs)
        #np.savetxt(f, tvecs)
        f.close()

        for i, img in enumerate(grabbedImgs):
            cv2.imwrite(os.path.join(dataDir, 'img{:3d}.png'.format(i)), img)


    # Closes all the frames
    cv2.destroyAllWindows()


if __name__=='__main__':
    main()