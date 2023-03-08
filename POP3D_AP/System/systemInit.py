import glob
import os
from FileOperations import loadVICONCalib
from FileOperations import rwOperations
from System import objectVicon
from Math import transformations as tf
from System import videoVicon
from System import imageVicon
import cv2 as cv
from System import camera
import numpy as np
import pickle

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



class SystemInit:

    def __init__(self, projectSettings, directoryName = None):
        """
        Author: Alex Chan
        VICON System init for custom cameras integration, skips steps that are related to vicon camera objects
        e.g vicon camera calibration parameters
        """
        self.settingsDict = projectSettings.settingsDict
        self.dataDir = self.settingsDict["DataDirectory"]

        self.sessionName = projectSettings.settingsDict["session"]

        self.rootDir = self.settingsDict["rootDirectory"]


        # Get information about camera objects
        self.customCameraInstances = []
        self.LoadCustomSonyCameras()

        self.sessionVideoFiles = self.generateVideoFileNames()

        # Load CSV file and create object
        self.CoordData = self.loadViconCoord() #load marker swap corrected 3D coordinates

        # Create objects for camera and objects
        self.viconObjects = self.loadVICONObjectsFromSubject()
        self.viconCameraObjects = self.loadVICONCameraObjects(customObjects=True)

        self.LoadSyncArray() #custom function to add sync array to object

        self.viconVideoObjects = self.loadSonyObjects()
        VideoDimension = (self.settingsDict["ResolutionWidth"],self.settingsDict["ResolutionHeight"])
        self.viconImageObjects = self.loadImageObjects(customObjectsDim=VideoDimension)

    def loadDataFile(self):
        print("Load data file")
        dataFileName = os.path.join(self.rootDirectory,self.settingsDict["dataFile"])
        # todo: videoToIRDataCaptureRatio should be saved in the settings file
        dataObject = rwOperations.TrackerDatabaseReader(dataFileName, self.settingsDict["objectsToTrack"])  # Read the csv file
        return dataObject
    
    def LoadSyncArray(self):
        DataDir = os.path.join(self.rootDir, "CalibrationInfo")

        SonyFirstFlash = []
        for i, cam in enumerate(self.settingsDict["cameras"]):
            SyncArrDir = os.path.join(DataDir, "%s-%s-SyncArray.p")%(self.sessionName,cam)
            SyncArray = pickle.load(open(SyncArrDir,"rb"))
            self.viconCameraObjects[i].SyncArray = SyncArray
            SonyFirstFlash.append(SyncArray[0][0])

        # import ipdb; ipdb.set_trace()

        #Get frame diff compared to earliest flash to synchronize all sony cams
        FrameDiff = [min(SonyFirstFlash)-x for x in SonyFirstFlash] #staggered start frames for vid

        for i in range(len(self.settingsDict["cameras"])):
            self.viconCameraObjects[i].FrameDiff = FrameDiff[i]
        # import ipdb; ipdb.set_trace()

    def loadViconCoord(self):
        print("Load marker swap corrected Matlab output csv")
        dataFileName = os.path.join(self.dataDir,"Raw3DTrackingData.csv")
        # todo: videoToIRDataCaptureRatio should be saved in the settings file
        dataObject = rwOperations.MatlabCSVReader(dataFileName, self.settingsDict["Subjects"],self.settingsDict["objectsToTrack"])  # Read the csv file
        return dataObject

    def loadFinalFeat(self):
        """Load final output csv of all features and custom features"""
        dataFileName = os.path.join(self.rootDirectory,self.settingsDict["FinalFeatureCSV"])
        # todo: videoToIRDataCaptureRatio should be saved in the settings file
        dataObject = rwOperations.MatlabCSVReader(dataFileName)  # Read the csv file
        return dataObject

    def loadCustomTrackingFeaturesToTrackingObjects(self):
        """
        Load the custom tracking features to the given vicon object
        :return:
        """
        customFilePath = os.path.join(self.rootDirectory, self.settingsDict["customFeatureFile3D"])
        for trackingObject in self.viconObjects:
            trackingObject.readFeaturePointsFromFile(customFilePath)
        return trackingObject


    def addCustomCameraInstances(self,
                                 type = "custom",
                                 id= 666,
                                 rot = [0, 0, 0, 1],
                                 translation = [0, 0, 0],
                                 k = np.identity(3),
                                 dist= np.zeros((1,5))
                                 ):
        """
        Creating the default custom camera instance
        :param type: string : Name of the camera
        :param id: string : Custom id given to the camera
        :param rot: list : Rotation
        :param translation: list : Translation
        :param k: list : Intrinsic parameters
        :param dist: list : Distortion parameters
        :return: None
        """
        print("No of custom cameras attached: ", len(self.customCameraInstances))
        cameraInstance = camera.Camera()
        cameraInstance.setCameraInfo(type,id)
        cameraInstance.setExtrinsicParam(rot, translation)
        cameraInstance.setIntrinsicParam(k, dist)
        self.customCameraInstances.append(cameraInstance)
        print("Added 1 camera instance: ", len(self.customCameraInstances))


    def loadVICONObjects(self):
        print("Load objects classes")
        viconObjects = []
        for object in self.settingsDict["ObjectID"]:
            viconObject = objectVicon.ObjectVicon(name=object)
            objPath = os.path.join(self.rootDirectory, object + ".mp")
            viconObject.readFeaturePointsFromFile(objPath)
            viconObjects.append(viconObject)

        return viconObjects

    def LoadCustomSonyCameras(self):
        """
        Author: Alex Chan
        Custom function to load all intrinsic/ extrinsic parameters from sony camera synced system.
        If you would like to add your own cameras, you will need you own function here
        """

        for i, cam in enumerate(self.settingsDict["cameras"]):
            DataDir = self.rootDir + "/CalibrationInfo/"
            # import ipdb;ipdb.set_trace()
            IntrinsicDir = os.path.join(DataDir ,"%s-%s-Intrinsics.p"%(self.sessionName,cam))
            ExtrinsicDir = os.path.join(DataDir ,"%s-%s-Extrinsics.p"%(self.sessionName,cam))
            # SyncArrDir = DataDir + cam + "_" +Video + "_SyncArr.p"

            rvec, tvec = pickle.load(open(ExtrinsicDir,"rb"))
            cameraMatrix, distCoeffs= pickle.load(open(IntrinsicDir,"rb"))
            # SyncArray = pickle.load(open(SyncArrDir,"rb"))

            # rot = rvec.reshape(1,3).tolist()[0]
            #convert rotation vector from opencv to rot matrix?0
            # import ipdb; ipdb.set_trace()
            rotMat = cv.Rodrigues(rvec)[0]
            rotQuat = tf.rotMatrixToQuat(rotMat) #trying quaternion
            rotList = [rotQuat[0],rotQuat[1],rotQuat[2],rotQuat[3]]


            trans = tvec.reshape(1,3).tolist()[0]

            self.addCustomCameraInstances("custom",i,rotList,trans,cameraMatrix,distCoeffs)
            # self.customCameraInstances[i].SyncArray = SyncArray


    def loadVICONObjectsFromSubject(self):
        """
        Author: Alex Chan
        Updated function to read seperate objects within each subject .mp file
        From different subject definition protocols
        """
        objects = self.settingsDict["objectsToTrack"]

        ViconObjects = []

        for object in objects:
            subject = object.split("_")[0] + "_" + object.split("_")[1]
            subPath = os.path.join(self.dataDir, subject + ".mp") #path to subject .mp file

            viconObject = objectVicon.ObjectVicon(name=object,subject = subject,object = object)
            viconObject.readFeaturePointsFromFileSubject(path = subPath)
            ViconObjects.append(viconObject)

        return ViconObjects
    
    def loadSonyObjects(self):
        """ Loading videos from custom sony camera system and naming conventions"""
        # Create video instances
        videoObjects = []

        for i in range(len(self.settingsDict["cameras"])):
            CamNum = i+1
            videoObject = videoVicon.VideoVicon(self.sessionVideoFiles[str("Cam%i"%CamNum)], int(CamNum))
            videoObjects.append(videoObject)

        return videoObjects

    def loadVideoObjects(self):
        # Create video instances
        videoObjects = []

        for serialNo in self.settingsDict["cameras"]:
            videoObject = videoVicon.VideoVicon(self.sessionVideoFiles[str(serialNo)], int(serialNo))
            videoObjects.append(videoObject)

        return videoObjects

    def loadImageObjects(self, customObjectsDim = False):

        imageObjects = []
        if customObjectsDim != False:
            cameraInstances = self.customCameraInstances
        else:
            cameraInstances = self.cameraInstances

        for cam in cameraInstances:
            # Create image instances, to store camera param and image features
            camParam = cam.getCameraParam()
            intrinsicMat = camParam["intrinsicParam"]
            distortionMat = camParam["distortionParam"]
            serialNo = camParam["serialNo"]
            if customObjectsDim == False:
                imageObjects.append(imageVicon.ImageVicon(serialNo, distortionMat, intrinsicMat))
            else:
                imageObjects.append(imageVicon.ImageVicon(serialNo, distortionMat, intrinsicMat,customObjectsDim))

        return imageObjects

    def loadVICONCameraObjects(self, customObjects = False):
        """
        Load camera objects
        :return: list of cam objects
        """
        viconCamObjects = []

        if customObjects:
            cameraInstances = self.customCameraInstances
        else:
            cameraInstances = self.cameraInstances

        for camera in cameraInstances:
            # Go through the instances and create camera object just like vicon object
            camParam = camera.getCameraParam()
            # Create camera object and add features to camera object from all possible vicon objects
            # The rotation parameters given in the calibration file are given to manipulate data from vicon space to camera space
            # We designed a new class called Object, this class is designed to store rotation and translation information
            # to transfer features to the vicon space
            rotation = camParam["extrinsicRotation"]
            # import ipdb; ipdb.set_trace()
            if customObjects:
                inverseRotation = rotation #If camera are custom, dont inverse
            else:
                inverseRotation = tf.invertQuaternion(rotation)
                
            viconCamObject = objectVicon.ObjectVicon(inverseRotation, camParam["extrinsicTranslation"],
                                                     camParam["serialNo"])
            print("Rotation: ", inverseRotation, "\n Translation: ", camParam["extrinsicTranslation"])

            for viconObject in self.viconObjects:  # Add features from each object to the camera object
                viconCamObject.setFeatures(viconObject.__getattribute__("featureDict"))

            viconCamObjects.append(viconCamObject)

        return viconCamObjects

    def verifyPath(self, path):
        """
        Verify the validity of the path otherwise return exception
        :param path: str
        :return: bool
        """
        return os.path.exists(path)

    def filterObjectBasedOnVideoFile(self):
        """
        Remove camera objects not having video files to support video tracking
        :return:
        """
        updatedCamObjects = []
        for i in range(len(self.cameraInstances)):
            if str(self.cameraInstances[i].__getattribute__("cameraID")) in self.sessionVideoFiles:
                updatedCamObjects.append(self.cameraInstances[i])

        return updatedCamObjects

    def setCalibFileName(self, name):
        """
        Sets custom file location for the calibration file. *.xcp
        :param name: File Name
        :return: None
        """
        self.calibFileName = name

    def setDataFileName(self,name):
        """
        Sets custom file location for the data file. *.csv
        :param name: File Name
        :return: None
        """
        self.dataFileName = name

    def printSysteInfo(self):
        """
        Prints all the information about the VICON system, obtained for the given file
        :return: None
        """
        print("File Path: ", self.rootDirectory)
        print("Session Name : ", self.sessionName)
        print("Data File Name", self.dataFileName)
        print("Calib File Name", self.calibFileName)
        print("Video Files", self.sessionVideoFiles)

    def generateCalibFileName(self):
        # Generate file name for the calibration file from the given info
        calibFileName = os.path.join(self.rootDirectory, self.settingsDict["calibFile"])
        # print(calibFileName)
        # Verification of Data
        if not os.path.exists(calibFileName):
            raise ValueError(" Error while generating *.xcp name using VICON convention, Check given Session name or File does not exist ")

        return calibFileName



    def loadCalibInfo(self):
        """
        Reads the calibration information from the given .xcp file
        :return: list of instances of class Camera
        """
        camObjects = loadVICONCalib.readCalibrationFromXCP(self.calibFileName)
        return camObjects

    # def getCameraCalibration(self):
    #     """
    #     Returns list of objects of class camera, read from the given .xcp file
    #     :return: list of Object camera
    #     """
    #     return self.camObjects

    def generateVideoFileNames(self):
        """
        Generates video files based on the given location of the file and the session name
        :return: Dict (Serial No : Video File)
        """
        sessionVideoFiles = {}
        for camera, videoFile in zip (self.settingsDict["cameras"], self.settingsDict["videoFiles"]):
            filePath = os.path.join(self.rootDir, videoFile)
            sessionVideoFiles[str(camera)] = filePath

        return sessionVideoFiles

    def generate2DAnnotationFileNames(self):
        """
        !! Edited to fit custom camera file structure of not everything being in same place!!

        Generate file names for the 2D annotation data for each camera,
        :return: Dict (Serial No: *.csv file)
        """
        assert (len(self.sessionVideoFiles) != 0),"No video files, can not generate annotation file names"
        annotationFilesDict = {}
        # import ipdb; ipdb.set_trace()
        for camera, annotationFileName in zip(self.settingsDict["cameras"],self.settingsDict["annotationFiles"]):
            filePath = os.path.join(self.dataDir,annotationFileName)
            annotationFilesDict[str(camera)] = filePath

        return annotationFilesDict





if __name__ == '__main__':
    # In this section we pass the arguments required for the main function to perform any operation.
    # Mostly it is the system information for calibration and the location of the VICON file

    settingsFile = "D:\\BirdTrackingProject\\20190618_PigeonPostureDataset\\settings_session02.xml"

    from FileOperations import settingsGenerator
    projectSettings = settingsGenerator.xmlSettingsParser(settingsFile)
    viconSystemData = VICONSystemInit(projectSettings)
    viconSystemData.printSysteInfo()
    # Get all the VICON camera instances
    camInstances = viconSystemData.cameraInstances
    printCalibInformation(camInstances)

    # create name for the annotation file
    dict = viconSystemData.generate2DAnnotationFileNames()

    # OR another way to access the information is this way
    print( " Camera type : " ,camInstances[0].cameraType)
    print(" Camera type : ", camInstances[0].extrinsicRotation)