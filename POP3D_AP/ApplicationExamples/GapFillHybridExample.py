"""
Author: Alex Chan

Application example of using DLC predictions as a gap filling mechanism for lost tracking

"""
import sys
sys.path.append('./')

from System import systemInit as system
from POP3D_AP.System import SettingsGenerator
from Math import stereoComputation as stereo
import random
import pandas as pd
import numpy as np
import math
from tqdm import tqdm
from glob import glob
import os



def Get_Dict(df,BBoxdf, frameNo, imsize=(320,320)):
    """Get feature dict from DLC predict input"""

    dfRow = df.loc[df.index=="frame-%i.jpg"%frameNo]
    Keypoints = list(set(list(df.columns.get_level_values(1))))
    Index = list(set(list(df.columns.get_level_values(0))))[0]
    featureDict = {}

    if len(dfRow) ==0:
        for key in Keypoints:
            featureDict.update({key:[0,0]})
        return featureDict,False
    else:

        # import ipdb;ipdb.set_trace()
        x_offset = round(BBoxdf.loc[frameNo,"452_0107_BBox_x"])
        y_offset = round(BBoxdf.loc[frameNo,"452_0107_BBox_y"])
        height = round(BBoxdf.loc[frameNo,"452_0107_BBox_h"])
        width = round(BBoxdf.loc[frameNo,"452_0107_BBox_w"])


        #get padding info:
        if width %2 ==0:
            #Shape is even number
            XPadLeft = int((imsize[0] - width)/2)
        else:
            XPadLeft =   int(((imsize[0] -width)/2)-0.5)
        
        if height %2 ==0:
            #Shape is even number
            YPadTop = int((imsize[1] - height)/2)
        else:
            YPadTop =int( ((imsize[1] -height)/2)-0.5)
        ####
        for key in Keypoints:
            featureDict.update({key:[dfRow[Index,key,"x"].to_list()[0]+x_offset-XPadLeft,dfRow[Index,key,"y"].to_list()[0]+y_offset-YPadTop]})
    
        # import ipdb;ipdb.set_trace()
        return featureDict, True #return boolean to show if there is data or not

def countGap(a):
    """Count nan gaps"""
    tmp = 0
    GapList = []
    for i in range(len(a)):
        current=a[i]
        if not(np.isnan(current)) and tmp>0:
            GapList.append(tmp)
            tmp=0
        if np.isnan(current):
            tmp=tmp+1
    return GapList
        

def IntroduceGap(CSV3Ddf):
    #Introduce NA to df:
    GapDF = CSV3Ddf.copy()
    GapDF = GapDF.iloc[350:,]
    GapDF = GapDF.dropna()
    TotalRows = len(GapDF)
    RowstoRemove = TotalRows * 0.3

    for i in range(int(RowstoRemove/30)):
        GapStartIndex = random.randint(0,TotalRows) #choose a random position to introduce 1 sec gap
        GapDF.loc[GapStartIndex:GapStartIndex+30,GapDF.columns != "Cam1frame"] = math.nan

    # countGap(GapDF['452_0107_bp_leftShoulder_z'].to_list())
    return GapDF



def FillGapWithTriangulation(settingsDict,CSV3DdfGap,FrameswithNA):

    # List of frame num that has NA:

    ##Read in data:
    ViconPath = settingsDict["ViconDirectory"]
    viconSystemData = system.VICONSystemInitCustom(projectSettings,ViconPath)
    viconCamObjects = viconSystemData.viconCameraObjets
    imageObjects = viconSystemData.viconImageObjects

    DLCModelName = "DLC_resnet50_20221108-163446_b4_max1500_opt-adam_m-resnet_5020221108-163446shuffle1_30000"
    PredictionPathList = sorted(glob("TempGapFillData/"+"*%s*"%DLCModelName))
    DLCdfList = []
    for path in PredictionPathList:
        DLCdfList.append(pd.read_hdf(path))

    BBoxpathList = [os.path.join(settingsDict["rootDirectory"],"%s_Cam%i.csv"%(settingsDict['FinalFeatureCSVBBox'],(i+1))) for i in range(4)]
    BBoxdfList = []
    for path in BBoxpathList:
        BBoxdfList.append(pd.read_csv(path))

    stereoTriangulator = stereo.StereoTrinagulator(viconCamObjects, imageObjects)

    #Gap fill df for DLC:
    DLCGapFilldf = CSV3DdfGap.copy()
    DLCGapFilldf["Views"] = math.nan
    viewsList = []
    #fill gap with triangulation
    for frame in tqdm(FrameswithNA):
        # if frame == 439:
        #     import ipdb;ipdb.set_trace()

        RealFrame = frame + viconCamObjects[0].FrameDiff #get actual frame relative to frame diff of cam 1, from frame diffs computed from desynced start times
        #get 2D data and set as feature:
        featureDictListBool = []
        for i in range(len(viconCamObjects)):
            TempframeNo = RealFrame - viconCamObjects[i].FrameDiff
            frameNo = BBoxdfList[i].loc[BBoxdfList[i]["frame"]==TempframeNo].index[0]
            viconCamObjects[i].clearFeatures()
            imageObjects[i].clearFeatures()

            ##get feature dict:
            featureDict, bool = Get_Dict(DLCdfList[i],BBoxdfList[i],frameNo)

            imageObjects[i].setFeatures(featureDict)
            featureDictListBool.append(bool)

        # print("%i views: %i"%(frameNo,sum(featureDictListBool)))
        RowIndex = (DLCGapFilldf["Cam1frame"] == frame).tolist().index(True)

        DLCGapFilldf.loc[RowIndex,"Views"] = sum(featureDictListBool)
        if sum(featureDictListBool)<2:#if only 1 view or no data at all
            continue

        triangulatedDict = stereoTriangulator.TriangulatePointsNViewBA(custom=True)
        
        for key, value in triangulatedDict.items():
            DLCGapFilldf.loc[RowIndex,"452_0107_%s_x"%key] = value[0]
            DLCGapFilldf.loc[RowIndex,"452_0107_%s_y"%key] = value[1]
            DLCGapFilldf.loc[RowIndex,"452_0107_%s_z"%key] = value[2]
            
    # DLCGapFilldf.iloc[RowIndex]
    # CSV3Ddf.iloc[RowIndex]
    return DLCGapFilldf

def ProcessFrameDict(Dict):
    """Provess frame wise dict to 3d dict per keypoint"""
    Dict.pop("Cam1frame")
    OutDict = {}
    counter = 0
    Point3D =[None]*3

    for key,val in Dict.items():
        if not val:
            Point3D[counter] = math.nan
        else:
            Point3D[counter] = list(val.values())[0]
        counter +=1

        if counter ==3:
            Name = key.strip("_z")
            OutDict[Name] = Point3D
            Point3D = [None]*3
            counter = 0
            # KeypointNameList.append(key.strip("_z"))
    
    return OutDict

def Calc_edErr_3D(pt1,pt2):
    """Calculate euclidian error between 2 3D points"""
    return math.sqrt(((pt2[0]-pt1[0])**2) +((pt2[1]-pt1[1])**2)+((pt2[2]-pt1[2])**2) )


def Get_MeanError(GroundTruthDF, df,FrameswithNA):
    """Get average error of all keypoints accross all NA frames"""
    AllErrorDict = {}

    for frame in tqdm(FrameswithNA):
        GTDict = GroundTruthDF.loc[GroundTruthDF["Cam1frame"]==frame].to_dict()
        DFDict = df.loc[df["Cam1frame"]==frame].to_dict()

        GTDict3D = ProcessFrameDict(GTDict)
        DFDict = ProcessFrameDict(DFDict)
        ##Calc average error:
        ErrorList = []
        for key in GTDict3D.keys():
            Error = Calc_edErr_3D(GTDict3D[key],DFDict[key])
            ErrorList.append(Error)
        # import ipdb;ipdb.set_trace()
        AvgError = sum(ErrorList)/len(ErrorList)
        AllErrorDict[frame] = AvgError

    return AllErrorDict

def Get_MeanError_Views(GroundTruthDF, df,FrameswithNA,ViewsCol):
    """Get average error of all keypoints accross all NA frames"""
    AllErrorDict = {}

    for frame in tqdm(FrameswithNA):
        GTDict = GroundTruthDF.loc[GroundTruthDF["Cam1frame"]==frame].to_dict()
        DFDict = df.loc[df["Cam1frame"]==frame].to_dict()
        Views = ViewsCol.loc[df["Cam1frame"]==frame,"Views"].to_list()[0]

        GTDict3D = ProcessFrameDict(GTDict)
        DFDict = ProcessFrameDict(DFDict)
        ##Calc average error:
        ErrorList = []
        for key in GTDict3D.keys():
            Error = Calc_edErr_3D(GTDict3D[key],DFDict[key])
            ErrorList.append(Error)
        # import ipdb;ipdb.set_trace()
        ##GET RMSE Errors:
        RMSEError = np.sqrt(np.mean(np.array(ErrorList)**2))
        AvgError = sum(ErrorList)/len(ErrorList)
        AllErrorDict[frame] = {"Error":AvgError, "RMSE":RMSEError,"Views":Views}
    

def Get_AllKeypointsError(GroundTruthDF, df,FrameswithNA):
    """Get error of all keypoints accross all NA frames"""
    AllErrorDict = {}

    for frame in tqdm(FrameswithNA):
        GTDict = GroundTruthDF.loc[GroundTruthDF["Cam1frame"]==frame].to_dict()
        DFDict = df.loc[df["Cam1frame"]==frame].to_dict()
        # Views = ViewsCol.loc[df["Cam1frame"]==frame,"Views"].to_list()[0]
        # import ipdb;ipdb.set_trace()

        GTDict3D = ProcessFrameDict(GTDict)
        DFDict = ProcessFrameDict(DFDict)
        ##Calc average error:
        # ErrorList = []
        ErrorDict = {}
        for key in GTDict3D.keys():
            Error = Calc_edErr_3D(GTDict3D[key],DFDict[key])
            ErrorDict[key] = Error
            # ErrorList.append(Error)
        # import ipdb;ipdb.set_trace()
        ##GET RMSE Errors:
        # RMSEError = np.sqrt(np.mean(np.array(ErrorList)**2))
        # AvgError = sum(ErrorList)/len(ErrorList)
        AllErrorDict[frame] = ErrorDict

    return AllErrorDict

if __name__ == "__main__":

    settingFile = "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/01072022Pigeon10/Annotation3DSony/settings_training_01072022_02.xml"
    SonyVideoName = "C0003"
    TrackedSubjects = [1]


    projectSettings = SettingsGenerator.xmlSettingsParser(settingFile,SonyVideoName,TrackedSubjects)
    settingsDict = projectSettings.settingsDict

    #original full dataframe:
    CSV3Ddf = pd.read_csv("TempGapFillData/training_01072022_02_Camfps3DKeypoint.csv")
    # CSV3DdfGap = IntroduceGap(CSV3Ddf)
    # CSV3DdfGap.to_csv("TempGapFillData/01072022_02_IntroducedNA.csv")
    # CSV3DdfGap["452_0107_bp1_x"].isna().sum()/len(CSV3DdfGap)
    CSV3DdfGap = pd.read_csv("TempGapFillData/01072022_02_IntroducedNA.csv")
    FrameswithNA = CSV3DdfGap["Cam1frame"].loc[CSV3DdfGap["452_0107_bp_leftShoulder_x"].isnull()].tolist()
    AllFrames = CSV3DdfGap["Cam1frame"].to_list()
    #remove Marker points:
    CSV3Ddf = CSV3Ddf[CSV3Ddf.columns.drop(list(CSV3Ddf.filter(regex="Unnamed")))]
    CSV3Ddf = CSV3Ddf[CSV3Ddf.columns.drop(list(CSV3Ddf.filter(regex="bp\d")))]
    CSV3Ddf = CSV3Ddf[CSV3Ddf.columns.drop(list(CSV3Ddf.filter(regex="hd\d")))]


    # DLCGapFilldf =  FillGapWithTriangulation(settingsDict,CSV3DdfGap,AllFrames)
    DLCGapFilldf =  FillGapWithTriangulation(settingsDict,CSV3DdfGap,FrameswithNA)
    #save views data
    # ViewsCol = DLCGapFilldf[["Views","Cam1frame"]]
    DLCGapFilldf = DLCGapFilldf[DLCGapFilldf.columns.drop(list(DLCGapFilldf.filter(regex="Views")))]

    #Filter out marker points:
    #remove "Unnamed Column"
    DLCGapFilldf = DLCGapFilldf[DLCGapFilldf.columns.drop(list(DLCGapFilldf.filter(regex="Unnamed")))]

    ##ErrorDF:
    ErrorDict = Get_AllKeypointsError(CSV3Ddf, DLCGapFilldf,FrameswithNA)
    ErrorDF = pd.DataFrame.from_dict(ErrorDict, orient = "index")
    ErrorDF = ErrorDF.dropna()

    print("DLC Gap Filling:")
    for col in ErrorDF.columns:
        data = ErrorDF[col].to_numpy()
        print("RMSE Error for " + col)
        print(np.sqrt(np.mean(data**2)))
    DLCGapFilldf.to_csv("TempGapFillData/GapFilledData.csv")
    # DLCGapFilldf.to_csv("TempGapFillData/FullDLCData.csv")

    
    
    ##fill gap with interpolation
    InterpolGapdf = CSV3DdfGap.copy()
    InterpolGapdf = InterpolGapdf.interpolate("linear")
    InterpolGapdf = InterpolGapdf[InterpolGapdf.columns.drop(list(InterpolGapdf.filter(regex="Unnamed")))]

    #error between original and interpolation
    ErrorDict2 = Get_AllKeypointsError(CSV3Ddf,InterpolGapdf,FrameswithNA)
    ErrorDF2 = pd.DataFrame.from_dict(ErrorDict2, orient = "index")
    ErrorDF2 = ErrorDF2.dropna()
    print("Linear Interpolation Gap Filling:")
    for col in ErrorDF2.columns:
        data = ErrorDF2[col].to_numpy()
        print("RMSE Error for " + col)
        print(np.sqrt(np.mean(data**2)))



    ##Kalman Filter?
    # from TempGapFillData import interpolation_5 as inter
    # KalmanGapdf = CSV3DdfGap.copy()
    # FrameCol = KalmanGapdf["Cam1frame"]
    # KalmanGapdf = KalmanGapdf[KalmanGapdf.columns.drop(list(KalmanGapdf.filter(regex="Unnamed")))]
    # KalmanGapdf = KalmanGapdf[KalmanGapdf.columns.drop(list(KalmanGapdf.filter(regex="hd\d")))]
    # KalmanGapdf = KalmanGapdf[KalmanGapdf.columns.drop(list(KalmanGapdf.filter(regex="bp\d")))]

    # KalmanGapdf = KalmanGapdf[KalmanGapdf.columns.drop(list(KalmanGapdf.filter(regex="Cam1frame")))]
    # KalmanGapFilledDF = pd.DataFrame()

    # while KalmanGapdf.shape[1]>1:
    #     df_ = KalmanGapdf.iloc[:, :3]
    #     df_ = inter.kalman_interpolate(df_)
    #     KalmanGapdf.drop(KalmanGapdf.iloc[:,:3], axis = 1, inplace=True)
    #     KalmanGapFilledDF = pd.concat([KalmanGapFilledDF, df_], axis=1)


    # KalmanGapFilledDF["Cam1frame"] = FrameCol
    # #Error:
    # ErrorDict3 = Get_AllKeypointsError(CSV3Ddf,KalmanGapFilledDF,FrameswithNA)
    # ErrorDF3 = pd.DataFrame.from_dict(ErrorDict3, orient = "index")
    # ErrorDF3 = ErrorDF3.dropna()
    # print("Kalman Filter Gap Filling:")
    # for col in ErrorDF3.columns:
    #     data = ErrorDF3[col].to_numpy()
    #     print("RMSE Error for " + col)
    #     print(np.sqrt(np.mean(data**2)))



    




