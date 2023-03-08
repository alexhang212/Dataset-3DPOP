"""
The application is test code for projecting 3D points on the image space using VICON data file.
The VICON output from Tracker is directly imported in the file and it can project the information from object space to image space.
Also from VICON space to image space.
The necessary steps are written in the file.

Note* : Initial code was written in jupyter notebook
"""
"""
As of 19.06.2019 : The file is "DEPRECATED". The file is going to be upgraded to visualise the posture information from the vicon system with the coordinate systems. 

"""

import cv2 as cv
import numpy as np
import pandas as pd
import glob
import os
from Math import transformations as tf
from Math import imageOperations as imgOp
from FileOperations import loadVICONCalib, rwOperations
from DrawingOperations import *

#Get names for loading the dataset
dirName = "D:/BirdTrackingProject/20180820_BirdTest"
sessionName = "20180820_BirdTest"
csvFile = os.path.join(dirName,"testData_Quaternion.csv")
videoFiles = glob.glob( os.path.join(dirName , "*.avi"))
calibFile = glob.glob(os.path.join(dirName , "*.xcp"))
objectPointsFile = os.path.join(dirName,"testbird_883.mp")


print("Video File Path : ", videoFiles)
print("Calibe File Path : ", calibFile)
print("csv File : ", csvFile)

# Given camera image undistort the image based on given parameters
calibObjects = []
videoFileName = []

# Read the calibration file
if len(calibFile) == 1:
    print("Calibration File Path : ", calibFile[0])
    camObjects = loadVICONCalib.readCalibrationFromXCP(calibFile[0])
else:
    raise ValueError("More than one .xcp files available")

# Read 3D point data
data = pd.read_csv(csvFile, float_precision='high')  # Read the csv file

# display window settings
windowName = "testVideo"
cv.namedWindow(windowName, cv.WINDOW_NORMAL)

# Get the setting for each camera
for camera in camObjects:

    # Go through each camera objects
    camera.printCameraParam()
    cameraParameters = camera.getCameraParam()
    intrinsicmatrix = cameraParameters["intrinsicParam"]
    distortionMatrix = cameraParameters["distortionParam"]
    cameraExtrinsicRot = cameraParameters["extrinsicRotation"]
    cameraExtrinsicTrans = cameraParameters["extrinsicTranslation"]
    camID = cameraParameters["serialNo"]

    for video in videoFiles:
        if str(camera.cameraID) in video:  # Check if camera ID is in video filename, take camera specific video
            videoFileName = video

    # Read the required video file
    cap = cv.VideoCapture(videoFileName)
    prevPoint = [0, 0]
    totalFrameCount = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    noData = False
    pause = False
    waitTime = 1

    while (True):
        #   try :
        # If viewer is paused we change the frame to be always at previous frame
        # if pause :
        # cap.set(cv.CAP_PROP_POS_FRAMES,frameCount - 1)
        frameCount = cap.get(cv.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()
        # Reading frame and marking
        if frame is not None:
            # TODO : Resolve Frame Number Ambiguity , cv.CAP_PROP_POS_FRAMES gives next frame number
            # Find corresponding row is csv : video frame rate (50 Hz) is half of vicon framerate (100 Hz)
            dataCount = frameCount*2 + 1 # Video Frame (n) = Data frame (2*n)+1 i.e. 0,1,2,3 - 1,3,5,7
            #dataCount = 2*frameCount + 2 # Video Frame n = Data frame (2*n)+2 i.e. 0,1,2,3 - 2,4,6,8

            print("Frame Count : ", frameCount, " / ", cap.get(cv.CAP_PROP_FRAME_COUNT))
            print("Data Count : ", dataCount)
            dataExists = []
            objectRotation = []
            objectTranslation = []
            #  image projections
            imgPoint = []
            bBoxPointsImg = []
            axisPointsImgSpace = []
            markerPointImgSpace = []

            if dataCount in data["Frame"].values:  # Check if the required frame exists in databased
                subData = data[data["Frame"] == dataCount]  # get required dataseries of that frame.
                dataSeries = subData.iloc[0, :]
                # Get the length of data series to determine how many objects exist in stream
                dataSeriesLength = dataSeries.size
                noOfObjects = int((dataSeriesLength - 2) / 7)
                print("No of Objects :", noOfObjects)
                objectRotation, objectTranslation = rwOperations.getPointData(dataSeries,
                                                                              noOfObjects)  # get rotation, translation of object

                # Check if the data has valid numbers
                for rot, trans in zip(objectRotation, objectTranslation):
                    print("Rotation : ", rot, "Translation :", trans)
                    if np.any(np.isnan(rot)) or np.any(np.isnan(trans)):  # if no valid info
                        # print("Tracking lost for this frame : ", frameCount)
                        dataExists.append(False)  # Save the data status
                    else:
                        dataExists.append(True)  # Save the data status
                print('Data : ', dataExists)
                # if no info in .csv
            else:
                print("Data Missing for frame : ", frameCount)
                continue

            # Go through each point and project it on image space
            # Point transferred from VICON subject space -> VICON space -> Camera space -> image space
            for point in range(len(dataExists)):

                if dataExists[point]:
                    # point in VICON subject space, we take origin of the object
                    position = [0, 0, 0]
                    boundingBoxPoints = generatePointsOp.getBoundingBox(position)
                    coordinatePoints = generatePointsOp.getCoordinatePoints()
                    markerPoints = generatePointsOp.getMarkerPoints(objectPointsFile)

                    print("Axis points Object Space : ", coordinatePoints)
                    # Debug msg
                    print("Obj Rot : ", objectRotation[point], ", Translation : ", objectTranslation[point])

                    # Object space to World
                    positionWorld = tf.transformPoint3D(position, objectRotation[point], objectTranslation[point])
                    bBoxPoints = tf.transformPoint3D(boundingBoxPoints, objectRotation[point], objectTranslation[point])
                    axisPoints = tf.transformPoint3D(coordinatePoints, objectRotation[point], objectTranslation[point])
                    markerPointsWorld = tf.transformPoint3D(markerPoints, objectRotation[point],
                                                            objectTranslation[point])
                    # Debug Msg
                    print("Axis points VICON Space : ", axisPoints)

                    # World space -> camera space
                    positionCamera = tf.worldToCameraSpace(positionWorld, cameraExtrinsicRot, cameraExtrinsicTrans)
                    bBoxCamSpace = tf.worldToCameraSpace(bBoxPoints, cameraExtrinsicRot, cameraExtrinsicTrans)
                    axisPointsCamSpace = tf.worldToCameraSpace(axisPoints, cameraExtrinsicRot, cameraExtrinsicTrans)
                    markerPointCamSpace = tf.worldToCameraSpace(markerPointsWorld, cameraExtrinsicRot,
                                                                cameraExtrinsicTrans)

                    # Debug Msg
                    print("Axis points Cam Space : ", axisPointsCamSpace)

                    # Camera space 3D -> 2D Image space
                    print("Position Camera: ", positionCamera)
                    imgPoint = imgOp.projectPointCamSpaceToImgSpace(positionCamera, intrinsicmatrix, distortionMatrix)
                    imgText = "Valid"
                    print("Projected Point : ", imgPoint)

                    bBoxPointsImg = imgOp.projectPointCamSpaceToImgSpace(bBoxCamSpace, intrinsicmatrix,
                                                                         distortionMatrix)
                    # print("bBoxPointsImg image space : ", bBoxPointsImg)

                    axisPointsImgSpace = imgOp.projectPointCamSpaceToImgSpace(axisPointsCamSpace, intrinsicmatrix,
                                                                              distortionMatrix)
                    # Debug Msg
                    print("AxisPoints image space : ", axisPointsImgSpace)

                    markerPointImgSpace = imgOp.projectPointCamSpaceToImgSpace(markerPointCamSpace, intrinsicmatrix,
                                                                               distortionMatrix)


                else:  # If point does not exist
                    imgpt = [0, 0]  # Fake point position on the image
                    imgPoint.append(imgpt)
                    imgText = "No Point"

            if frameCount == totalFrameCount:
                print("Cutting off @ frame : ", frameCount)
                break

            # HYR =(Height, Y coordinate, Rows), WXC = (Width, X coordinate, Cols)
            height, width, channel = frame.shape

            # Undistorting for plotting points
            undistortedImage = cv.undistort(frame, intrinsicmatrix, distortionMatrix)

            if point in range(len(imgPoint)):
                # If point in valid i.e. in the image.
                if imgOp.isPointValid(imgPoint[point], height, width):
                    print("Image point : VALID")
                    cv.circle(undistortedImage, (int(imgPoint[point][0]), int(imgPoint[point][1])), 10, (255, 255, 0), 1)
                    if len(axisPointsImgSpace):
                        drawOp.drawCoordinateAxis(undistortedImage, axisPointsImgSpace)

                    if len(markerPointImgSpace):
                        drawOp.drawMarkerPoints(undistortedImage, markerPointImgSpace, 2)

                    if len(bBoxPointsImg):
                        print("bBoxPointsExist")
                        minIndex, maxIndex = imgOp.filterBBoxPoints(bBoxPointsImg)
                        cv.rectangle(undistortedImage,
                                      (int(bBoxPointsImg[minIndex][0]), int(bBoxPointsImg[minIndex][1])),
                                      (int(bBoxPointsImg[maxIndex][0]), int(bBoxPointsImg[maxIndex][1])), (255, 0, 0),
                                      10)

                        # Drawing functions
                        # drawBoundingBox(undistortedImage, bBoxPointsImg)

                    else:
                        print("Doesnt exist")

                    font = cv.FONT_HERSHEY_SIMPLEX
                    cv.putText(undistortedImage, str(frameCount) + "_Point_" + str(point) + imgText, (10, 50), font, 1,
                               (255, 255, 255), 2, cv.LINE_AA)

                else:  # point invalid
                    imgText = "Out of Image"
                    cv.circle(undistortedImage, (0, 0), 20, (0, 0, 255), 1)
                    cv.putText(undistortedImage, str(frameCount) + "_Point_" + str(point) + imgText, (10, 50), font, 1,
                               (255, 255, 255), 2, cv.LINE_AA)
                    # print("Image point : INVALID")

            cv.imshow(windowName, undistortedImage)
            # For exiting the loop
            k = cv.waitKey(waitTime)
            if k == ord('q'):
                break
            if k == ord('x'):
                cap.set(cv.CAP_PROP_POS_FRAMES, frameCount + 1000)
            if k == ord("b"):
                cap.set(cv.CAP_PROP_POS_FRAMES, frameCount - 1)
                continue
            if k == ord("p"):
                # pause = not pause
                waitTime = 0
            if k == ord("r"):
                waitTime = 1
        else:  # If frame does not exist
            print("Exit Code")
            break

cap.release()
cv.destroyAllWindows()