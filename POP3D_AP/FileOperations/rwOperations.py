# Read and Writing related helper functions for dealing with data series
import numpy as np
import pandas as pd
import os
import re

def readFeaturesFromFile(path):
    """
    Loads features from the given file into a dictionary
    :param  : Path to text file with all required features
    :return : return list of features
    """
    features = []
    assert (os.path.exists(path) == True), "File for custom features does not exist"
    assert (".txt" in path), "Given feature file is not in .txt format"

    file = open(path)
    lines = [line.rstrip('\n') for line in file]
    # Read point information from the file
    for line in lines:
        featureName = line.split("=")
        features.append(str(featureName[0]))

    return features

def readFeaturePointsFromViconFileAsArray(path):
    """
    The function reads point features from vicon file and returns as array 3xN
    :param path: Location of the .mp file as string
    :return: returns point as 3xN array for direct manipulations
    """
    featureDict = readFeaturePointsFromViconFile(path)
    pointArray = np.array([featureDict[i] for i in featureDict])
    rows, cols = pointArray.shape
    if rows != 3 and cols == 3:
        pointArray = pointArray.T
    else:
        raise Exception(
            "Given point array is not upto specification A is not 3xN or Nx3, it is {}x{}".format(rows, cols))

    return pointArray

"""TODO: Refacture name that gives idea of return type e.g. readFeaturePointsFromViconFileAsDict"""
def readFeaturePointsFromViconFile(path):
    """
    Reads the vicon files for objects .mp and converts into dict of features with 3D coordinates
    :param path of .mp files as str
    :return: dictionary of 3D points
    """
    assert (os.path.exists(path) == True), "File: does not exist"

    file = open(path)
    lines = [line.rstrip('\n') for line in file]
    points = []
    # Read point information from the file
    for line in lines:
        pointInfo = line.split("=")
        if (len(pointInfo) == 2):
            points.append(float(pointInfo[1]))

    name = 1
    objectBasename = os.path.basename(path)
    objectBasename = objectBasename.split(".")[0]

    featureDict = {}
    # Store pattern information as list
    for i in range(0, len(points), 3):
        pt = [points[i], points[i + 1], points[i + 2]]
        featureDict[objectBasename + str(name)] = pt
        name += 1
    # End loop
    return featureDict

def readFeaturePointsFromViconFileSubjects(path, object = ["hd"], ObjectLen = 4):
    """
    Author: Alex Chan
    New function to read .mp files for given subject, which invludes multiple objects
    :param path: str of input path
    :param objects: list of object names to look for within the .mp file
    :param ObjectLen: length of object, default 4
    :return: dictionary of 3D points
    """
    assert (os.path.exists(path) == True), "File: does not exist"

    file = open(path)
    lines = [line.rstrip('\n') for line in file]
    points = []
    pointNames = []
    # Read point information from the file
    for line in lines:
        pointInfo = line.split("=")
        if (len(pointInfo) == 2):
            pointNames.append(pointInfo[0])
            points.append(float(pointInfo[1]))

    objectBasename = os.path.basename(path)
    objectBasename = objectBasename.split(".")[0] #subject name
    # objectBasename = objectBasename.split(".")[1]#objectBasename

    featureDict = {}
    ObjectType = object.split("_")[2]

    for i in range(ObjectLen):
        ObjName = ObjectType+str(i+1)
        MatchIndex = [j for j in range(len(pointNames)) if re.search(r'%s'%ObjName,pointNames[j])]
        if len(MatchIndex)==3: #check if length is 3, should always be 3
            pt = [points[MatchIndex[0]], points[MatchIndex[1]], points[MatchIndex[2]]]
            featureDict[objectBasename + "_" + ObjName] = pt
        else:
            raise Exception("there are more than 3 objects of name %s in %s"%(ObjName,path))
    # import ipdb; ipdb.set_trace()

    return featureDict
                

def get3DDictofObject(frameDict, object, ObjectLen = 4):
    """
    Author: Alex Chan
    Given dictionary of 3D points for a given frame and object name, 
    extract all the 3D points for that object

    :param frameDict: dictionary of 3D points of a given frame
    :param object: str name of object
    :param ObjectLen: length of object, assume default 4
    :return: dict of the points  
    """
    DictNames = list(frameDict.keys())
    MatchIndex = [k for k in range(len(DictNames)) if re.search(r'%s'%object,DictNames[k])]
    PtDict = {}
    counter = 0

    if len(MatchIndex) == ObjectLen*3:
        for m in range(ObjectLen):
            pts = [frameDict[DictNames[MatchIndex[counter]]],frameDict[DictNames[MatchIndex[counter+1]]],frameDict[DictNames[MatchIndex[counter+2]]]]
            PtDict.update({"%s%i"%(object,m+1):pts})
            counter += 3
    return PtDict


def readFeaturePointsFromTextFile (path):
    assert (os.path.exists(path) == True), "File does not exist"

    file = open(path)
    lines = [line.rstrip('\n') for line in file]
    featureDict = {}
    # Read point information from the file
    for line in lines:
        pointInfo = line.split("=")
        if (len(pointInfo) == 2):
            point = pointInfo[1].split(",")
            featureDict[pointInfo[0]] = [float(point[0]) , float(point[1]) , float(point[2])]
        else:
            featureDict = {}
            return featureDict

    return featureDict

def writeFeaturePointsToFile(path, featureDict):
    """

    :param path:
    :param featureDict:
    :return:
    """
    assert (".txt" in path), "File for custom features does not exist"

    file = open(path,'w')

    for feature in featureDict:
        x = featureDict[feature][0]
        y = featureDict[feature][1]
        z = featureDict[feature][2]

        file.write(feature + "=" + str(x) + "," + str(y) + "," + str(z) + "\n")

    file.close()

    return True

def writeFeaturePointsToFileSubject(path, featureDict, subject):
    """
    Author Alex Chan
    Write custom feature point to txt for each inidivual subject
    :param path:
    :param featureDict:
    :return:
    """
    assert (".txt" in path), "File for custom features does not exist"

    file = open(path,'w')

    for feature in featureDict:
        x = featureDict[feature][0]
        y = featureDict[feature][1]
        z = featureDict[feature][2]

        file.write(feature + "=" + str(x) + "," + str(y) + "," + str(z) + "\n")

    file.close()

    return True


def createEmptyDictFromList(featureList, pointInfo = "2D"):
    """
    Create default values given feature list
    :param featureList: list of features
    :param pointInfo: str : "2D" or "3D"
    :return:
    """
    dict = {}
    assert( pointInfo == "2D" or pointInfo == "3D"), "Point info Error."
    assert (len(featureList) != 0)," Empty feature list given"

    for feature in featureList:
        if pointInfo == "2D":
            dict[feature]= [0,0]
        elif pointInfo == "3D":
            dict[feature] = [0, 0, 0]

    return dict

class annotationDatabase:
    """
    The class is designed to create annotation file consisting of frame info, 2D annotation and 3D annotation, bBox annotation
    """
    def __init__(self, path, featureList, resetPreviousAnnotations = False):
        """
        Create database class
        :param path:
        :param featureList:
        """
        self.path = path
        self.featureList = featureList
        self.featureDict2D = {}
        self.featureDict3D = {}
        self.featureDictBbox = {}

        self.defaultDataSeries = self.generateDataSeries() # Create empy data series
        if os.path.exists(self.path):
            # If file exists we have to read the file from database
            self.dataBase = pd.read_csv(self.path)
        else:
            #if file does not exists we have to create it
            self.dataBase = pd.DataFrame(columns=list(self.defaultDataSeries))

        if resetPreviousAnnotations:
            self.dataBase = self.dataBase.iloc[0:0]

    def saveDataBase(self): # Saving database to csv file
        """
        Save the database to .csv format
        :return: bool
        """
        self.dataBase.to_csv(self.path, index = False)

        return True

    def updateDataSeries(self, dataSeries, dict, pointInfo = "2D"):
        """
        Update the dataSeries for the given frame, by loading the dictionary in right format
        :param dataSeries: dict
        :param dict: dict (To be saved)
        :param pointInfo: 2D or 3D point
        :return: bool
        """
        if pointInfo == "2D":
            for feature in self.featureList:
                if feature in dict: # if the required feature exists in the dict
                    dataSeries[feature+"_2d_x"] = dict[feature][0]
                    dataSeries[feature+"_2d_y"] = dict[feature][1]
        elif pointInfo == "3D":
            for feature in self.featureList:
                if feature in dict:
                    dataSeries[feature+"_3d_x"] = dict[feature][0]
                    dataSeries[feature + "_3d_y"] = dict[feature][1]
                    dataSeries[feature + "_3d_z"] = dict[feature][2]
        elif pointInfo == "bBox":
            for feature in dict:
                dataSeries[feature + "_2d_x"] = dict[feature][0]
                dataSeries[feature + "_2d_y"] = dict[feature][1]

        return True


    def createFeatureDict(self, featureList, pointInfo = "2D"):
        """
        Create feature dictionary based on the given features and point info, empty dictionary
        :param featureList: list : of features to be used for creating dict
        :param pointInfo: str
        :return: dict
        """
        defaultDataSeriesDict = {}
        if pointInfo == "2D":
            for feature in featureList:
                defaultDataSeriesDict[str(feature) + "_2d_x"] = 0
                defaultDataSeriesDict[str(feature) + "_2d_y"] = 0
        elif pointInfo == "3D":
            for feature in featureList:
                defaultDataSeriesDict[str(feature) + "_3d_x"] = 0
                defaultDataSeriesDict[str(feature) + "_3d_y"] = 0
                defaultDataSeriesDict[str(feature) + "_3d_z"] = 0
        else:
            defaultDataSeriesDict = {}

        return defaultDataSeriesDict

    def generateDataSeries(self):
        """
        Create .csv file based on the given feature List
        :return:
        """
        defaultDataSeries = {"frame": 0}
        self.featureDict2D = self.createFeatureDict(self.featureList, "2D")
        defaultDataSeries.update(self.featureDict2D)
        self.featureDict3D = self.createFeatureDict(self.featureList, "3D")
        defaultDataSeries.update(self.featureDict3D)
        self.featureDictBbox = self.createFeatureDict(["lCorner","rCorner"])
        defaultDataSeries.update(self.featureDictBbox)

        return defaultDataSeries

    def updateDataBase(self, frameNo, featureDict2D, featureDict3D, bBoxDict):
        """
        Update database based on the given information
        :return:
        """
        dataSeries = self.defaultDataSeries.copy() # create a clone
        dataSeries["frame"] = frameNo

        self.updateDataSeries(dataSeries, featureDict2D, "2D")
        self.updateDataSeries(dataSeries, featureDict3D, "3D")
        self.updateDataSeries(dataSeries, bBoxDict, "bBox")

        # If exists
        if frameNo in self.dataBase["frame"].tolist():
            index = self.dataBase["frame"].tolist().index(frameNo)
            self.dataBase.loc[index] = dataSeries
        else:  # if entry does not exist we add a new
            self.dataBase = self.dataBase.append(dataSeries, ignore_index=True)

        self.saveDataBase()

        return True

    def readDataFrame(self, data):
        """
        Read the given data frame and save the feature in the feature dictionary
        :return:
        """
        featureDict2D = {}
        featureDict3D = {}
        featureDictBBox = {}
        for feature in self.featureList:
            featureDict2D[feature] = [data[feature + "_2d_x"].values[0],data[feature + "_2d_y"].values[0]]
            featureDict3D[feature] = [data[feature + "_3d_x"].values[0],data[feature + "_3d_y"].values[0],data[feature + "_3d_z"].values[0]]

        for feature in self.featureDictBbox:
            featureDictBBox[feature] = data[feature].values[0]

        return featureDict2D,featureDict3D,featureDictBBox

    def getDataFromVideoFrame(self, videoFrameNo):
        assert (videoFrameNo >= 0), " Query frame no less than 0. Check!!"
        dataFrame = self.dataBase[self.dataBase["frame"] == videoFrameNo]
        if not dataFrame.empty:
            return self.readDataFrame(dataFrame)
        else:
            return self.generateDataSeries()

class ImageAnnotationDatabaseReader:
    """
    The class is designed to read the annotation database for the given 2D features and return the features.
    """
    def __init__(self, path , featureList):
        """
        Contrusctor for reading the features from the annotation files
        :param pathDict: str
        :param features: dict
        """
        self.dataBaseFile = path
        self.featureList = featureList
        self.featuresDict = self.createDict()
        self.data = pd.read_csv(self.dataBaseFile)

    def createDict(self):
        """
        Create default values for the provided features
        :return: dict
        """
        return createEmptyDictFromList(self.featureList, "2D")

    def readDataFrame(self, data):
        """
        Read the given data frame and save the feature in the feature dictionary
        :return:
        """
        for feature in self.featuresDict:
            self.featuresDict[feature] = [data[feature + "_x"].values[0],data[feature + "_y"].values[0]]
        return True


    def getDataForVideoFrame(self, videoFrameNo):
        """
        Get data for given video frame from the .csv annotation file.
        :param int
        :return: dictionary
        """
        assert (videoFrameNo >=0)," Query frame no less than 0. Check!!"
        dataFrame = self.data[self.data["frame"] == videoFrameNo]
        if not dataFrame.empty:
            self.readDataFrame(dataFrame)
            return self.featuresDict
        else:
            return self.createDict()

class NexusDatabaseReader:
    """
    The class is designed to load the database created by nexus software and return the 3D features for the given objects
    """
    def __init__(self, filePath, featureList, videoToIRDataCaptureRatio = 0.5, readOutStartIndex = 0):
        """
        Constructor for the class to read the database created by nexus software,
        :param filePath: str - path to file .csv
        :param featureList: list - list of features to be read from file
        :param videoToIRDataCaptureRatio: float - videoCaptureRate(FPS)/viconCaptureRate(FPS)
        :param readOutStartIndex: int - offset between first frame of video and first frame of vicon
        """
        assert (os.path.exists(filePath)),"Given Nexus data base file does not exist, Check path"
        self.dataBaseFile = filePath

        assert (len(featureList) != 0), "No features to read"
        self.featureList = featureList
        self.featureDict = self.createDict()

        data = pd.read_csv(self.dataBaseFile, header = 3, skiprows=[4])
        self.dataFrame = self.filterData(data)

        self.videoCameraFrameRateRatio = videoToIRDataCaptureRatio
        self.startIndex = 0

    def filterData(self, data):
        """
        Returns data structure with frames which have at least some tracking data, i.e. removes holes from .csv file
        :return: data frame (pandas)
        """
        filteredData = data[data["Sub Frame"]==0]
        return filteredData

    def createDict(self):
        """
        Create keys for the data frame from the given feature list
        :return: Empty dict with zero values
        """
        return createEmptyDictFromList(self.featureList,"3D")

    def computeDataFrameNoFromVideoFrameNo(self, videoFrameNo):
        """
        Computes the corresponding data frame number using given video frame number
        :param videoFrameNo: video frame no for which data is required
        :return: data frame no, the frame number registered by VICON system
        """
        assert (videoFrameNo >= 0), "Video frame number can not be less than 0"

        dataFrameNo = ( videoFrameNo * int(1/self.videoCameraFrameRateRatio) ) + 1 + self.startIndex
        return np.floor(dataFrameNo)

    def getDataForVideoFrame(self, videoFrameNumber):
        """
        returns rotation, translation data for the query video frame
        :param videoFrameNumber: query frame
        :return: list
        """
        dataFrameNumber = self.computeDataFrameNoFromVideoFrameNo(videoFrameNumber)
        #print("Video Frame:",videoFrameNumber, " -- dataFrame no", dataFrameNumber)

        if self.getFrameData(dataFrameNumber) is True:
            return self.featureDict
        else:
            return self.createDict()


    def getFrameData(self, dataFrameNumber):
        """
        provides data for the query frame number (vicon tracking)
        :param dataFrameNumber: query frame
        :return: dict
        """

        if not self.dataFrame[self.dataFrame["Frame"] == dataFrameNumber].empty:
            dataList = self.dataFrame[self.dataFrame["Frame"] == dataFrameNumber].values[0].tolist()
            subDataList = dataList[2:]

            if len(self.featureDict)*3 != len(subDataList):
                print("Feature and column list of data do not meet")
                return False

            for feature,index in zip(self.featureDict,range(len(self.featureDict))):
                xData = subDataList[3*index + 0]
                yData = subDataList[3*index + 1]
                zData = subDataList[3*index + 2]
                self.featureDict[feature] = [xData, yData, zData]

            return True

        else:
            print("Given frame is empty!! Readout error")
            return False


class MatlabCSVReader:
    #Class to read csv output from matlab program developed by Kano
    #Author: Alex Chan

    def __init__(self,fileName,subjects= None, objectsToTrack=None,videoToIRDataCaptureRatio=0.5):
        """
        Initializes matlab csv database, stores XYZ coordinates of points in vicon coordinate system
        of each tracked subject (e.g Bird) and each object (e.g head + backpack)
        :param fileName str: path of file
        :param subjects list: list of strings, of the subjects in the trial
        :param objectsToTrack: list of strings, of objects within each subject to track
        """
        assert (os.path.exists(fileName)), "Matlab CSV file does not exist, Check path"
        self.dataBaseFile = fileName

        self.objectsToTrack = objectsToTrack
        self.subjects = subjects

        self.data = self.ReadData(self.dataBaseFile)

        # VICON tracking frame rate rate is higher than camera
        self.videoCameraFrameRateRatio = videoToIRDataCaptureRatio

    def ReadData(self, fileName):
        """Reads in data from file"""
        df = pd.read_csv(fileName, header=0)
        return df

    def getCoordDataForVideoFrame(self, frameNum):
        """
        Function to get data for given camera frame
        """
        #first get frame in dataframe based on frame rate
        dfFrame = ((1/self.videoCameraFrameRateRatio)*frameNum)-1 #minus one due to indexing
        ExtractedRow = dict(self.data.loc[dfFrame,:])
        return ExtractedRow

    def getCoordDataForViconFrame(self, frameNum):
        """
        Function to get data for given vicon time step, without correcting for frame rate
        """
        #first get frame in dataframe based on frame rate
        dfFrame = frameNum
        # import ipdb; ipdb.set_trace()
        Index = self.data.index[self.data["Frame"]==frameNum].to_list()[0]

        ExtractedRow = dict(self.data.loc[Index,:])
        return ExtractedRow

    def getFrameData(self, dataFrameNumber, FeatureList):
        """
        provides data for the query frame number
        :param dataFrameNumber: query frame
        :return: dict
        """

        #for different names of frame for different CSVs
        if 'frame' in self.data.columns:
            FrameName = 'frame'
        elif 'Frame' in self.data.columns:
            FrameName = 'Frame'

        dataFrameNumber = ((1/self.videoCameraFrameRateRatio)*dataFrameNumber)
        OutDict=dict.fromkeys(FeatureList)
    
        if not self.data[self.data[FrameName] == dataFrameNumber].empty:
            dataList = self.data[self.data[FrameName] == dataFrameNumber].values[0].tolist()
            subDataList = dataList[1:]

            if len(FeatureList)*3 != len(subDataList):
                print("Feature and column list of data do not meet")
                # return False

            for feature,index in zip(FeatureList,range(len(FeatureList))):
                xData = subDataList[3*index + 0]
                yData = subDataList[3*index + 1]
                zData = subDataList[3*index + 2]
                OutDict[feature] = [xData, yData, zData]

            return OutDict

        else:
            print("Given frame is empty!! Readout error")
            return False






class TrackerDatabaseReader:
    # The class used to manage all the information regarding a frame
    def __init__(self, fileName, objectsToTrack, videoToIRDataCaptureRatio = 0.5, readOutStartIndex = 0):
        """
        Initialize the database class, which stores the transformation information about the vicon objects
        :param fileName: str : name of file
        :param objectsToTrack: list of objects to track
        :param videoToIRDataCaptureRatio: Vicon Frame Rate / Video frame rate
        :param readOutStartIndex: Frame mapping between 1st frame of vicon and video
        """

        assert (os.path.exists(fileName)), "Given Tracker data base file does not exist, Check path"
        self.dataBaseFile = fileName

        self.objectsToTrack = objectsToTrack

        assert (len(self.objectsToTrack) != 0), "No features to read"
        self.objectParamDict = self.createDict()

        # data = pd.read_csv(name, float_precision='high') # old code reads clean file
        data = pd.read_csv(self.dataBaseFile, header=3, skiprows=[4]) # reads direct output from tracker
        self.data = self.filterData(data) # Remove rows for which we have no data from VICON (i.e. Sub Frame != 0)

        # VICON tracking frame rate rate is higher than camera
        self.videoCameraFrameRateRatio = videoToIRDataCaptureRatio
        self.startIndex = readOutStartIndex

    def checkDataValidity(self, objectRotation, objectTranslation):
        """
        Check if the given rotation and translation parameters are valid
        :param objectRotation: List of fortation parameters
        :param objectTranslation: List of translation parameters
        :return: True or False
        """

        assert(len(objectTranslation) != 0 and len(objectRotation) != 0), "Given rotation, trasnlation parameter list are empty"

        if np.any(np.isnan(objectTranslation)) or np.any(np.isnan(objectRotation)):
            return False
        else:
            return True


    def getPointData(self, objectData):
        """
        The function gets a line from .csv file, reads rotation and translation data from the given data series
        and separates the rotation and translation data based on number of objects
        :param objectData: rot, trans data for all objects -> Dataseries Pandas
        :param noOfObjects: no of objects
        :return: ndarry list of [ rot, trans ] object wise separation
        """
        noOfObjects = len(self.objectsToTrack)
        # Save rotation and translation information for given objects
        for i in range(noOfObjects):
            # Rotation and Translation data is from column 2-7.
            rot = [objectData[2 + (7 * i) + 0], objectData[2 + (7 * i) + 1], objectData[2 + (7 * i) + 2],
                   objectData[2 + (7 * i) + 3]]
            # rot = [ angle*180/np.pi for angle in rot]
            trans = [objectData[2 + (7 * i) + 4], objectData[2 + (7 * i) + 5], objectData[2 + (7 * i) + 6]]
            self.checkDataValidity(rot,trans)
            self.objectParamDict[self.objectsToTrack[i] + "_rotation"] = rot
            self.objectParamDict[self.objectsToTrack[i] + "_translation"] = trans
            self.objectParamDict[self.objectsToTrack[i] + "_validity"] = self.checkDataValidity(rot,trans)

        return True

    def createDict(self):
        """
        Create dictionary for rotation translation parameters for the given Object
        :param ObjectsToTrack:
        :return: dict
        """
        objectParamDict = {}
        for object in self.objectsToTrack:
            objectParamDict[object + "_rotation"] = [1,0,0,0]
            objectParamDict[object + "_translation"] = [0,0,0]
            objectParamDict[object + "_validity"] = False

        return objectParamDict

    def dataSize(self):
        """
        number of data points
        :return: int number
        """
        return self.data.shape[0]

    def filterData(self, data):
        """
        Returns data structure with frames which have at least some tracking data, i.e. removes holes from .csv file
        :return: data frame (pandas)
        """
        filteredData = data[data["Sub Frame"]==0]
        return filteredData

    def computeDataFrameNoFromVideoFrameNo(self, videoFrameNo):
        """
        Computes the corresponding data frame number using given video frame number
        :param videoFrameNo: video frame no for which data is required
        :return: data frame no, the frame number registered by VICON system
        """
        assert (videoFrameNo >= 0), "Video frame number can not be less than 0"

        dataFrameNo = ( videoFrameNo * int(1/self.videoCameraFrameRateRatio) ) + 1 + self.startIndex
        return np.floor(dataFrameNo)

    def getDataForVideoFrame(self, videoFrameNumber):
        """
        returns rotation, translation data for the query video frame
        :param videoFrameNumber: query frame
        :return: rotation, translation, validity [list]
        """
        assert (videoFrameNumber >= 0), "Video frame number can not be less than 0."
        dataFrameNumber = self.computeDataFrameNoFromVideoFrameNo(videoFrameNumber)
        valid = self.getFrameData(dataFrameNumber)

        if not valid: # if values were not avialble, return default dictionary
            self.objectParamDict = self.createDict()

        return self.objectParamDict

    def getFrameData(self, dataFrameNumber):
        """
        provides data for the query frame number (vicon tracking)
        :param dataFrameNumber: query frame
        :return: rotation, translation , validity [list]
        """

        if dataFrameNumber in self.data["Frame"].values:  # Check if the required frame exists in databased
            subData = self.data[self.data["Frame"] == dataFrameNumber]  # OR
            dataSeries = subData.iloc[0]
            # Get the length of data series to determine how many objects exist in stream
            dataSeriesLength = dataSeries.size
            noOfObjects = int((dataSeriesLength - 2) / 7)
            assert(noOfObjects == len(self.objectsToTrack))," Object param mismatch, given object length does not match with object data in .csv"
            self.getPointData(dataSeries)  # get rotation, translation of object
            return True
        else:
            print("Frame data missing in .csv or video framerate ratio : ", dataFrameNumber)
            return False


class customFeatureGenerator:
    """
    The class to read the 3D file and figure out 3D points
    """
    def __init__(self, path, featureList):
        """

        :param settingsDict:
        """
        self.dataBaseFile = path
        self.featureList = featureList
        self.featuresDict = self.createDict()
        self.data = pd.read_csv(self.dataBaseFile)

        self.generate3DFeatures()

    def findOutliers(self, filteredList, avgPoint):
        distList = []

        for point in filteredList:
            dist = np.sqrt( np.power((point[0]-avgPoint[0]),2) + np.power((point[1]-avgPoint[1]),2) + np.power((point[2]-avgPoint[2]),2) )
            distList.append(dist)

        return distList

    def computeAvgPoint(self, xFeatures, yFeatures, zFeatures ):

        xPoint = []
        yPoint = []
        zPoint = []
        filteredList = []
        avgPoint = []
        for x,y,z in zip(xFeatures,yFeatures,zFeatures):
            if x != 0 and y != 0 and z != 0 :
                xPoint.append(x)
                yPoint.append(y)
                zPoint.append(z)
                filteredList.append([x,y,z])
        if len(xPoint) > 0 and len(yPoint) > 0 and len(zPoint) > 0:
            xAvg = sum(xPoint) / len(xPoint)
            yAvg = sum(yPoint) / len(yPoint)
            zAvg = sum(zPoint) / len(zPoint)
            avgPoint = [xAvg,yAvg,zAvg]

        distList = self.findOutliers(filteredList,avgPoint)
        return distList

    def generate3DFeatures(self):

        listOfFrames = self.data["frame"].tolist()
        for feature in self.featuresDict:
            xFeature = self.data[str(feature) + "_x"].tolist()
            yFeature = self.data[str(feature) + "_y"].tolist()
            zFeature = self.data[str(feature) + "_z"].tolist()
            distList = self.computeAvgPoint(xFeature,yFeature,zFeature)
            #print("Feature : ", feature, " - DistanceList : ", distList)

    def createDict(self):
        """
        Create default values for the
        :return: dict
        """
        return createEmptyDictFromList(self.featureList, "3D")

###################################################### ------------------- UNIT TEST ###############################


def unitTestTrackerDatabase():
    #fileName = "D:\\BirdTrackingProject\\20180906_1BirdBackpack\\20180906_1BirdBackpack_Bird201.csv"
    fileName = "20190131_pigeonVideo_2MarkerHead.csv"

    objectsToTrack = ["backpack", "head"]
    # Read dataset from the given .csv file
    viconDataObject = TrackerDatabaseReader(fileName, objectsToTrack)

    for i in range(0, 1000, 50):
        print(" Video Frame No :", i, " <-> Data Frame No : ", viconDataObject.computeDataFrameNoFromVideoFrameNo(i))
        objectParamDict = viconDataObject.getDataForVideoFrame(i)

        for trackingObject in objectsToTrack:
            print("Rotation", objectParamDict[trackingObject+"_rotation"])
            print("Translation" , objectParamDict[trackingObject+"_translation"])
            print("Validity", objectParamDict[trackingObject+"_validity"])


# todo : Protocol for reading nexus files have to be defined later, it has complexity of having both objects and features.
# How do we read and store such data is kind of questionable. For now it reads only objects
def unitTestNexusDataBase():

    file = "dataExamples/20190304_CalibSessionVicon09_nexus.csv"
    featureList = ["Hero1","Hero2","Hero3","Hero4"]

    nexusDataReader = NexusDatabaseReader(file, featureList)

    frameData = [0,4,6,10,14]
    for frameNo in frameData:
        featureDict = nexusDataReader.getDataForVideoFrame(frameNo)
        print(featureDict)

def unitTestImageAnnotationDataBase():

    file = "D:\BirdTrackingProject\VICON_DataRead\VICONFileOperations\dataExamples\\20190304_CalibSessionVicon_09.2118670.csv"
    featureFilePath = "D:\\BirdTrackingProject\\testDataSet\\customFeatures.txt"

    features = readFeaturesFromFile(featureFilePath)
    assert(os.path.exists(file)), "File does not exist"

    annotationReader = ImageAnnotationDatabaseReader(file, features)

    frameData = [0, 4, 6, 10, 14]

    for frameNo in frameData:
        featureDict = annotationReader.getDataForVideoFrame(frameNo)
        print(featureDict)

def unitTestAnnotationDataBase():
    file = "D:\\BirdTrackingProject\\20190618_PigeonPostureDataset\\20190618_PigeonPostureDataset_session02.2118670.database.csv"
    featureList = readFeaturesFromFile("D:\\BirdTrackingProject\\20190618_PigeonPostureDataset\\20190618_PigeonPostureDataset_session02.customFeatures.txt")
    dataBase = annotationDatabase(file, featureList)

    for i in range(0,320,10):
        dict2D,dict3D,dictBBox = dataBase.getDataFromVideoFrame(i)
        print("Dictionary 2D", dict2D)
        print("Dictionary 3D", dict3D)
        print("Bounding box :", dictBBox)

def unitTestCustomFeatureGenerator():
    file = "D:\\BirdTrackingProject\\20190618_PigeonPostureDataset\\20190618_PigeonPostureDataset_session02.3D.csv"
    featureList = readFeaturesFromFile("D:\\BirdTrackingProject\\20190618_PigeonPostureDataset\\20190618_PigeonPostureDataset_session02.customFeatures.txt")

    customFeatureGenerator(file,featureList)


def main():

    # path = "temp.txt"
    # dict = {"point1": [1.000, 2.334, 3],
    #         "point2": [4, 2.3, 3],
    #         "point3": [2, 2, 3.22],
    #         "point4": [1.1, 2.3, 3.222]}

    #writeFeaturePointsToFile(path,dict)
    #print(readFeaturePointsFromTextFile(path))
    print(readFeaturePointsFromViconFile(path="..\VICONTestData\object1.mp"))
    print(readFeaturePointsFromViconFileAsArray(path="..\VICONTestData\object1.mp"))

    #unitTestCustomFeatureGenerator()

    #unitTestAnnotationDataBase()

    #unitTestTrackerDatabase()

    #unitTestNexusDataBase()

    #unitTestImageAnnotationDataBase()

    # featureList = ["Hero1","Hero2","Hero3","Hero4"]
    # print("2D Features", createEmptyDictFromList(featureList,"2D"))
    # print("3D Features", createEmptyDictFromList(featureList, "3D"))


if __name__ == '__main__':
    main()