# The file is object for VICON. This is effective method for storing and
# accesing information about the vicon objects

from re import X
import numpy as np
import os
from Math import transformations as transferOp
from FileOperations import rwOperations
from Math import absoluteOrientation
import math
import pyquaternion as pq
import pandas as pd

def get_Dist(point1,point2):
    """Compute distance between 2 3D points"""
    dist = math.sqrt((point2[0]-point1[0])**2 + (point2[1]-point1[1])**2+(point2[2]-point1[2])**2)
    return dist

class ObjectVicon:
    """The class is object is base class for VICONTrackingObjects and VICONCameraObjects"""
    def __init__(self, rotation = [0, 0, 0, 1], translation = [0,0,0], name = "temp", subject = "temp",object="temp"):
        """
        Initialize the class, the position orientation and name of object
        :param rotation: list of roation parameters
        :param translation: list of translation parameters
        :param name: name of the object
        """
        self.rotation = rotation # list of parameters
        self.translation = translation # list of parameters
        self.name = name

        self.subject = subject
        self.object = object

        self.featureMat = np.zeros((3,1))
        self.featurePoints = []
        self.featureList = []

        self.featureDict = {}
        
        #REMOVE LATER
        # self.TEMPDATA = pd.read_csv("~/Desktop/452_0107_Rotation.csv")
        self.PrevFrame = [0,0,0]
        self.DistList = []

    def clearFeatures(self):
        """
        Clear all features
        :return: True
        """
        self.featureDict = {}

    def setTransformationParameters(self, rotation, translation ):

        self.rotation = rotation
        self.translation = translation

    def GetRotTransFromViconSpace(self,Points3D):
        """
        Author: Alex Chan
        Given 3D points from vicon space and object points from .mp file, 
        get rotation translation of whole object 
        """

        ##Check for cases of NAN in 3D points
        NADetect = [math.isnan(x[0]) for x in Points3D.values()]
        NAindex = np.where(NADetect)[0].tolist() #index of the points that are NA

        #prepare data
        ObjectFeat = []
        RealCoord = []

        featureDict = self.featureDict.copy()
    
        if len(NAindex)>0:
            return False, False
            #NA detected
            ##just return false
            
        ObjectFeatKey = list(featureDict.keys())
        RealCoordKey = list(Points3D.keys())

        for i in range(len(Points3D)):
            #find index where names match the object
            ObjectFeatMatch = [ObjectFeatKey[j] for j in range(len(ObjectFeatKey)) if ObjectFeatKey[j].endswith("%s%i"%(self.object,i+1))]
            RealCoordMatch = [RealCoordKey[j] for j in range(len(RealCoordKey)) if RealCoordKey[j].endswith("%s%i"%(self.object,i+1))]
            # print(ObjectFeatMatch)
            # print(RealCoordMatch)

            ObjectFeat.append(featureDict[ObjectFeatMatch[0]])
            RealCoord.append(Points3D[RealCoordMatch[0]])
        
            
        ObjectFeatMat = np.matrix(ObjectFeat).T
        RealCoordMat = np.matrix(RealCoord).T
        
        Rotation, Translation = absoluteOrientation.findPoseFromPoints(ObjectFeatMat,RealCoordMat,self.name)
        
        return Rotation, Translation

    def transferFeaturesToViconSpace(self, custom = False) :
        """
        Transfer the features from Object space to target space
        :return: dict (3D features)
        """
        assert (len(self.featureDict) != 0), "Dicitonary empty!! No object features to transfer"
        featureList = list(self.featureDict.values())
        if custom:
            targetPoints = transferOp.transformPoint3D(featureList, self.rotation, self.translation,False,custom=custom)
        else:
            targetPoints = transferOp.transformPoint3D(featureList, self.rotation, self.translation)

        
        targetPoints = targetPoints.tolist()
        transferedFeatureDict = {}
        for feature,index in zip(self.featureDict,range(len(targetPoints)) ):
            transferedFeatureDict[feature] = targetPoints[index]

        return transferedFeatureDict
    
    def transferFeaturesToViconSpaceSimple(self):
        """
        Author: Alex Chan
        Something weird with the function above with transformPoint3D, which uses quaternion
        temp function to test more simple transformation using 3x3 rotational matrix
        """
        # import ipdb; ipdb.set_trace()

        assert (len(self.featureDict) != 0), "Dicitonary empty!! No object features to transfer"
        featureList = np.matrix(list(self.featureDict.values())).T
        
        targetPoints = transferOp.transformPoints(featureList, self.rotation, self.translation)
        targetPoints = np.array(targetPoints).T #transpose
        targetPoints = targetPoints.tolist()
        transferedFeatureDict = {}
        for feature,index in zip(self.featureDict,range(len(targetPoints)) ):
            transferedFeatureDict[feature] = targetPoints[index]

        return transferedFeatureDict

    def transferFeaturesToObjectSpaceSimple(self,featureDictViconSpace, custom=False):
        """
        Author: Alex Chan
        Something weird with the function below with transformPoint3D, which uses quaternion
        temp function to test more simple transformation using 3x3 rotational matrix
        """
        assert (len(featureDictViconSpace) != 0), "Given featurelist is empty"
        rotationMatrix = self.rotation

        featureList = np.matrix(list(featureDictViconSpace.values())).T
        Inv_rotation= np.linalg.inv(rotationMatrix)
        # import ipdb; ipdb.set_trace()

        targetPoints = transferOp.invertPoints(featureList,Inv_rotation, self.translation)
        targetPoints = np.array(targetPoints).T #transpose
        targetPoints = targetPoints.tolist()
        transferedFeatureDict = {}
        for feature, index in zip(featureDictViconSpace, range(len(targetPoints)) ):
            transferedFeatureDict[feature] = targetPoints[index]

        return transferedFeatureDict


    def transferFeaturesToObjectSpace(self, featureDictViconSpace, custom=False):
        """
        Transfer feature points from given coordinate system to object space
        :return: dictionary of features
        """
        assert (len(featureDictViconSpace) != 0), "Given featurelist is empty"

        featureList = list(featureDictViconSpace.values())
        if custom:
            targetPoints = transferOp.transformPoint3D(featureList, self.rotation, self.translation, False, custom=True)
        else:
            targetPoints = transferOp.transformPoint3D(featureList, self.rotation, self.translation, True)
        
        
        targetPoints = targetPoints.tolist()
        transferedFeatureDict = {}
        for feature, index in zip(featureDictViconSpace, range(len(targetPoints)) ):
            transferedFeatureDict[feature] = targetPoints[index]

        return transferedFeatureDict

    def transferPointsToObjectSpace(self, pointList):
        """
        Transfers the list points from given space to Object space using given R and T
        :return:
        """
        invertedPoints = transferOp.transformPoint3D(pointList, self.rotation , self.translation, True) # R_inv . (P-T)
        return invertedPoints

    def transferPointsFromObjectSpace(self, pointList):
        """
        Transfers given list of points from object space to target space using given R and T
        :return:
        """
        targetPoints = transferOp.transformPoint3D(pointList,self.rotation,self.translation)
        return targetPoints

    def setFeatures(self, featuresDict):
        """
        Set the given features in the feature list
        :param  set the features in the dict using given dict
        :return: True/False
        """
        for feature in featuresDict:
            self.featureDict[feature] = featuresDict[feature]

        return True

    def removeSelectedFeatures(self, features):
        """
        Remove the given features
        :param featureDict: dict
        :return: bool
        """
        for feature in features:
            if feature in self.featureDict:
                del self.featureDict[feature]

        return True

    def filterFeaturesBasedOnObject(self, dict):
        """
        Filter the feature dict to only have features from a specific object
        :return: dict
        """
        modifiedDict = {}
        for feature in dict :
            if self.name in feature:
                modifiedDict[feature] = dict[feature]

        return modifiedDict

    def readFeaturePointsFromFile(self, path, objectFilter = True):
        """
        Reads given *.mp (VICON) file to create feature points for Object
        :param path: Storage path of the
        :return: True if success
        """
        if os.path.exists(path) and ".mp" in path:
            featureDict = rwOperations.readFeaturePointsFromViconFile(path)
            self.setFeatures(featureDict)
            return True

        elif os.path.exists(path) and ".txt" in path:
            featureDict = rwOperations.readFeaturePointsFromTextFile(path)
            if objectFilter:
                objectSpecificDict = self.filterFeaturesBasedOnObject(featureDict)

            self.setFeatures(objectSpecificDict)
            return True

        else:
            raise ValueError("Given file does not exist ", path)

    def readFeaturePointsFromFileSubject(self,path):
        """
        Author: Alex Chan
        Updated function to read .mp file based on target object and subject
        when multiple objects are present within a subject .mp file
        """

        if os.path.exists(path) and ".mp" in path:
            featureDict = rwOperations.readFeaturePointsFromViconFileSubjects(path, object = self.object)
            self.setFeatures(featureDict)
            return True
        else:
            raise ValueError("Given file does not exist ", path)
        


if __name__ == '__main__':

    rotation = [0 , 0 , 0 , 1] # Identity
    translation = [0 , 0 , 0 ] # Translation

    # Test creating points from file
    filePath = "D:/BirdTrackingProject/VICON_DataRead/VICONDrawingOperations\\9mm_02.mp"
    objectName = "9mm_02"
    viconObject = ObjectVicon(rotation, translation, objectName)
    viconObject.readFeaturePointsFromFile(filePath)
    viconObject.name = "point"
    viconObject.readFeaturePointsFromFile("D:\BirdTrackingProject\VICON_DataRead\VICONFileOperations\\temp.txt")
    print("Print features: ", viconObject.featureDict)
    # Test creating points without file
    pointlist = list(viconObject.featureDict.values())



    rotationCam1 = [-0.79697172135980876, 0.009835241003934278, 0.056076936965809787, 0.60132746530298287]
    translationCam1 = [-701.064504485933, -6173.2248621199, 1830.24808693825]

    cam1Points = [[1415.76135949, - 570.12878565, 7724.00698054],
                           [1427.61481598, - 588.07764951, 7708.65018457],
                           [1433.03420332, - 579.43874105, 7728.08209005],
                           [1439.77318296, - 598.69765411, 7698.82965911]]

    viconObject.setTransformationParameters(rotationCam1,translationCam1)

    # transfer points to object space
    transferedPoints = viconObject.transferPointsFromObjectSpace(cam1Points)
    print("TF Points:", transferedPoints)
    reversedPoints = viconObject.transferPointsToObjectSpace(transferedPoints)
    print("Reversed points:", reversedPoints)

    transferedDict = viconObject.transferFeaturesToViconSpace()
    print("TF Points:", transferedDict)
    reversedDict = viconObject.transferFeaturesToObjectSpace(transferedDict)
    print("Reversed points:", reversedDict)

    # Test adding feature and clear the features
    p = {'1': [0, 0, 0]}
    viconObject.setFeatures(p)
    print("Print added features: ", viconObject.featureDict)

    viconObject.removeSelectedFeatures(p)
    print("Print removed features: ", viconObject.featureDict)

    viconObject.clearFeatures()
    print("Print features after clearance: ", viconObject.featureDict)