"""
Author: Alex Chan

Part of the auto-keypoint annotation pipeline, takes 3D keypoints and reproject to 2D for each camera view

"""

from System import systemInit as system
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2 as cv
import pickle
import math
import multiprocessing as mp
from FileOperations import rwOperations


def createDatabase2D(customFeatures):
    """
    Create databse structure for storage of reprojected 2D features

    :return:
    """

    defaultDataSeries = {"frame": 0}
    for feature in customFeatures:
        defaultDataSeries[ str(feature) + "_x"] = 0
        defaultDataSeries[str(feature) + "_y"] = 0

    dataFrame = pd.DataFrame(columns=list(defaultDataSeries))

    return defaultDataSeries, dataFrame

def FindClosestFrame(SyncArray, Frame):
    """
    Given a sony frame and a SyncArray, find closest frame in vicon
    
    Will throw error if frame is before first flash
    """
    #Index of frame where it is last synced
    #if minus the current frame, smaller than 1 (i.e any negative number and 0)
    index = np.where((SyncArray[0]-Frame) <1)[0].argmax()
    #find proportion of how far the frame is within the gap
    Prop = (Frame - SyncArray[0][index])/(SyncArray[0][index+1]- SyncArray[0][index])

    #Calculate corresponding vicon frame from the proportion
    ViconFrame = SyncArray[1][index]+(Prop*(SyncArray[1][index+1] - SyncArray[1][index]))

    return ViconFrame

def GetObjectList(df):
    """Get list of all objects in a df"""
    dfCol = list(df.columns)
    dfCol.remove('frame')
    KeypointNameList = []

    for i in range(len(dfCol)):
        if (i+1) % 3 ==0:
            KeypointNameList.append(dfCol[i].strip("_z"))

    return KeypointNameList

def Generate2DKeypointCam(df, camMat, distCoef,rvec,tvec,FrameCount,SyncArray,imageWidth,imageHeight ):
        ##Loop through each frame and cam to reproject

    OutputDict = {}
    KeypointNameList = GetObjectList(df)

    for i in tqdm(range(SyncArray[0][0],SyncArray[0][len(SyncArray[0])-1])): #from first flash to end
        ViconFrame = round(FindClosestFrame(SyncArray, i))


        FrameData = df.loc[df["frame"]==ViconFrame].to_dict()
        FrameData.pop("frame")
        counter = 0
        Point3D =[None]*3
        # KeypointNameList = []
        Point3DList=[]
        
        for key,val in FrameData.items():
            if not val:
                Point3D[counter] = math.nan
            else:
                Point3D[counter] = list(val.values())[0]
            counter +=1

            if counter ==3:
                Point3DList.append(Point3D)
                Point3D = [None]*3
                counter = 0
                # KeypointNameList.append(key.strip("_z"))


        Points3DArr = np.float32(Point3DList).reshape(-1,3)

        #reproject points:
        TrialFeatureCoordDict, _ = createDatabase2D(KeypointNameList)

        All2DPoints, jac = cv.projectPoints(Points3DArr, rvec, tvec, camMat, distCoef)
        FrameDict = {KeypointNameList[x]:All2DPoints[x][0].tolist() for x in range(len(All2DPoints))}
        #filter out points that are outside frame:
        FrameDict = {key:([math.nan,math.nan] if val[0]>imageWidth or val[1]>imageHeight else val) for key,val in FrameDict.items()}
                
        for feature in FrameDict:
            point2D = FrameDict[feature]
            TrialFeatureCoordDict[feature + "_x"] = point2D[0]
            TrialFeatureCoordDict[feature+ "_y"] = point2D[1]

        TrialFeatureCoordDict["frame"] = i
        OutputDict.update({str(i):TrialFeatureCoordDict})

    return OutputDict

def Generate3DKeypoint_Camfps(df,SyncArray):
    """Extra function to run once for 3D data with camera fps"""


    OutDict3Dfps = {}

    for i in tqdm(range(SyncArray[0][0],SyncArray[0][len(SyncArray[0])-1])): #from first flash to end
        ViconFrame = round(FindClosestFrame(SyncArray, i))

        FrameData = df.loc[df["frame"]==ViconFrame].to_dict()
        FrameData.pop("frame")

        #for frame fps 3D data:
        FrameDataCopy = FrameData.copy()

        FrameDataCopy = {k:list(v.values())[0] for k,v in FrameDataCopy.items()}
        FrameDataCopy.update({"frame":i})
        OutDict3Dfps.update({i:FrameDataCopy})
    #save Cam fps 3D dict:
    Camfps3Ddf = pd.DataFrame.from_dict(OutDict3Dfps, orient = "index")

    return Camfps3Ddf


def GenerateKeypointCamLoop(i, df,viconSystemData,settingsDict,AnnotDir,basedir,imageWidth,imageHeight):
        Cam=i+1
        camMat = viconSystemData.viconImageObjects[i].intrinsicMatrix
        distCoef= viconSystemData.viconImageObjects[i].distortionMatrix

        #just using pickles to load extrinsics
        # import ipdb; ipdb.set_trace()
        ExtPath = os.path.join(basedir,"CalibrationInfo","%s-Cam%i-Extrinsics.p"%(settingsDict["session"],Cam))
        rvec, tvec = pickle.load(open(ExtPath,"rb"))
        FrameCount = viconSystemData.viconVideoObjects[i].totalFrameCount

        SyncPath = os.path.join(basedir,"CalibrationInfo","%s-Cam%i-SyncArray.p"%(settingsDict["session"],Cam))
        SyncArray = pickle.load(open(SyncPath,"rb"))
        # import ipdb;ipdb.set_trace()

        Camfps3Ddf = Generate3DKeypoint_Camfps(df, SyncArray)
        Cam3Dpath = os.path.join(AnnotDir,settingsDict["FinalFeatureCSV3D"][i])
        Camfps3Ddf.to_csv(Cam3Dpath)

        ##For 2D Keypoints:
        OutputDict = Generate2DKeypointCam(df, camMat, distCoef,rvec,tvec,FrameCount,SyncArray,imageWidth,imageHeight)

        outDF = pd.DataFrame.from_dict(OutputDict, orient="index")
        OutPath = os.path.join(AnnotDir,settingsDict["FinalFeatureCSV2D"][i])
        outDF.to_csv(OutPath, index=False)

        return 1

def Generate2DKeypoint(projectSettings):
    """Generate 2D Keypoints from 3D features for each camera view"""
    #load required files
    # import ipdb;ipdb.set_trace()
    settingsDict = projectSettings.settingsDict
    basedir = settingsDict["rootDirectory"]
    AnnotDir = settingsDict["AnnotationDirectory"]
    dataDir = settingsDict["DataDirectory"]
    
    viconSystemData = system.SystemInit(projectSettings)
    

    TrialFeatureCoordPath =  os.path.join(AnnotDir, settingsDict["FinalFeatureCSV"])
    imageHeight = int(viconSystemData.viconImageObjects[1].imageHeight)
    imageWidth = int(viconSystemData.viconImageObjects[1].imageWidth)


    #read in final features csv:

    df = pd.read_csv(TrialFeatureCoordPath)
    for i in range(4):
        GenerateKeypointCamLoop(i, df,viconSystemData,settingsDict,AnnotDir,basedir,imageWidth,imageHeight)

