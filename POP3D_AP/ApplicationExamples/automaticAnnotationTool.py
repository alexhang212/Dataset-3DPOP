"""
The automatic annotation tool works in following way. It takes custom 3D feature information of given 6DOF VICON objects
(i.e. marker position of marker patterns and virtual features prepared through annotation protocol) and projects them on the image.
The final projected points are stored in the .csv file to create a database for the image annotations.
"""

import cv2 as cv
from System import systemInit as system
from FileOperations import settingsGenerator
import os
from FileOperations import rwOperations

# Debug Function to print the calibration Information
def printCalibInformation(camInstances):
    for cam in camInstances:
        camera = cam.getCameraParam()
        print("intrinsicParam : " , camera["intrinsicParam"])
        print("extrinsicRotation : ", camera["extrinsicRotation"])
        print("extrinsicTranslation : ",camera["extrinsicTranslation"])
        print("distortionParam : ", camera["distortionParam"])
        print("Cam ID", camera["serialNo"])
        print("Camera Type", camera["cameraType"])

def getFeaturesInViconSpace(viconObjects,transformationParamDict):
    featureDictViconSpace = {}
    # Loop through objects, if tracked then transfer those feature from object space to vicon space
    for j in range(len(viconObjects)):
        objName = viconObjects[j].__getattribute__("name")
        if transformationParamDict[objName + "_validity"] == True:
            viconObjects[j].setTransformationParameters(transformationParamDict[objName + "_rotation"],
                                                        transformationParamDict[objName + "_translation"])

            featureDictViconSpace.update(viconObjects[j].transferFeaturesToViconSpace())

    return featureDictViconSpace

def getFeaturesInCameraSpace(viconCamObjects,featureDictViconSpace):

    for j in range(len(viconCamObjects)):  # Loop through camera objects and find image projections
        viconCamObjects[j].clearFeatures()
        # Dictionary operations
        featureDictCamSpace = viconCamObjects[j].transferFeaturesToObjectSpace(featureDictViconSpace)
        viconCamObjects[j].setFeatures(featureDictCamSpace)
        # print("CameraFeatures : ", featureDictCamSpace)
    return True

def getFeaturesInImageSpace(imageObjects,viconCamObjects):
    for j in range(len(viconCamObjects)):  # Loop through camera objects and find image projections
        imageObjects[j].clearFeatures()
        imageFeaturesDict = imageObjects[j].projectFeaturesFromCamSpaceToImageSpace(viconCamObjects[j].featureDict)
        imageObjects[j].setFeatures(imageFeaturesDict)
        # print("ImageFeatures : ", imageFeaturesDict)

def videoWriter(windowNames, FPS = 30, imageWidth = 1920, imageHeight = 1080):
    videoOut = []
    for i in range(len(windowNames)):
        name = windowNames[i] + "output.mp4"
        videoOut.append(cv.VideoWriter(name, 0x00000020, FPS, (imageWidth, imageHeight), True))  # 400,400
    return videoOut

def releaseVideos(videoOut):
    """
    Release the holder for video if video output is stored
    :param videoOut: videoOutput Objects
    :return: bool
    """
    for video in videoOut:
        video.release()

    return True

def main(settingsFile, writeVideo = False, showImages = False):

    projectSettings = settingsGenerator.xmlSettingsParser(settingsFile)
    directoryName = projectSettings.settingsDict["rootDirectory"]

    # Part 1 : Get system information
    """
    First part of software gets information about the current session, associated .csv files produced by vicon
    and calibration information for video cameras,
    """
    viconSystemData = system.VICONSystemInit(projectSettings,directoryName)
    # viconSystemData.printSysteInfo()

    # Part 2 : Read data frame and create vicon objects
    dataObject = viconSystemData.dataObject # fileOp.TrackerDatabaseReader(viconSystemData.dataFileName, objectsToTrack)  # Read the csv file
    viconObjects = viconSystemData.viconObjects

    # Create video objects
    videoFiles = viconSystemData.sessionVideoFiles #  __getattribute__("sessionVideoFiles")
    viconSystemData.loadCustomTrackingFeaturesToTrackingObjects()
    videoObjects = viconSystemData.viconVideoObjects
    viconCamObjects = viconSystemData.viconCameraObjets
    imageObjects = viconSystemData.viconImageObjects

    # Define window names
    windowNames = []
    maxFrameNo = []
    for video in videoObjects:
        windowNames.append(video.windowName)
        maxFrameNo.append(video.totalFrameCount)
        if showImages:
            cv.namedWindow(video.windowName, cv.WINDOW_NORMAL)

    # Check if all videos have same number of frames and reassigns the variable to one single value
    if maxFrameNo.count(maxFrameNo[0]) == len(maxFrameNo):
        maxFrameNo = maxFrameNo[0]

    # Video writing
    if writeVideo:
        videoOut = videoWriter(windowNames, FPS=30)

    dataBaseObjects = []
    featureFile = os.path.join(directoryName, projectSettings.settingsDict["customFeatureFile"])
    featureList = rwOperations.readFeaturesFromFile(featureFile)
    dataBaseFiles = projectSettings.settingsDict["annotationDataBaseFiles"]
    for dataBaseFile in dataBaseFiles:
        path = os.path.join(directoryName, dataBaseFile)
        dataBaseObjects.append( rwOperations.annotationDatabase(path, featureList, resetPreviousAnnotations= True) )

    # Part 3 : Process frame information ( Frame by Frame or All together)
    for i in range(0,maxFrameNo,2):

        print("Processing data for Frame No : ", i)
        # Get all rotation and translation
        transformationParamDict = dataObject.getDataForVideoFrame(i)
        featureDictViconSpace = getFeaturesInViconSpace(viconObjects,transformationParamDict)

        if len(featureDictViconSpace) == 0:
            print("No tracking data skip Projection for frame : ", i)
            continue

        getFeaturesInCameraSpace(viconCamObjects,featureDictViconSpace)
        getFeaturesInImageSpace(imageObjects,viconCamObjects)

        # Get image frame and draw the information on image space
        for j in range(len(videoObjects)):
            if showImages:
                tempImage = videoObjects[j].getFrame(i)
                if tempImage is not None:
                    #imageObjects[j].drawFeatures(image= tempImage, pointSize= 2)
                    imageObjects[j].drawPosture(tempImage)
                    imageObjects[j].drawFeatures(tempImage)
                    bBoxDict = imageObjects[j].drawBoundingBox(tempImage, border = 10)

                    roiImage = imageObjects[j].getRoiOnKeypoint("body_leftShoulder",tempImage)
                    cv.imshow(windowNames[j], tempImage)
                    if writeVideo:
                        videoOut[j].write(tempImage)
            bBoxDict = imageObjects[j].computeBoundingBox()
            dataBaseObjects[j].updateDataBase(i, imageObjects[j].featureDict, viconCamObjects[j].featureDict, bBoxDict)

        k = cv.waitKey(10)
        if k == ord('q'):
            cv.destroyAllWindows()
            if writeVideo:
                releaseVideos(videoOut)
            # Exit the program
            break

        if k == ord('n'):
            continue


if __name__ == '__main__':
    # In this section we pass the arguments required for the main function to perform any operation.
    # Mostly it is the system information for calibration and the location of the VICON file

    settingsFile = "D:\\BirdTrackingProject\\20190620_PigeonPostureDataset4\\settings_session07.xml"

    main(settingsFile, showImages= True)



