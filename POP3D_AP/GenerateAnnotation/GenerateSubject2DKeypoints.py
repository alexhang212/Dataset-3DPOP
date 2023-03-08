"""
Author: Alex Chan

Part of the auto-keypoint annotation pipeline,takes bounding boxes, crop to subject
then output mini video of subject through trial.

"""

from System import systemInit as system
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2 as cv
import pickle
import math
import multiprocessing as mp


def GenerateSubject2DKeypoints(projectSettings,imsize):
    """Generate videos and csv for each subject of keypoints"""

    settingsDict = projectSettings.settingsDict
    directoryName = settingsDict["rootDirectory"]

    if settingsDict["Mode"] == "Custom":
        ViconPath = settingsDict["ViconDirectory"]
        viconSystemData = system.VICONSystemInitCustom(projectSettings,ViconPath)
        custom = True
    else:
        sessionName = settingsDict["session"]
        viconSystemData = system.VICONSystemInit(projectSettings,sessionName)
        custom = False

    videoObjects = viconSystemData.viconVideoObjects
    
    if custom:
        for i in range(4):
            Cam = i+1
            BBoxPath = os.path.join(directoryName,settingsDict["FinalFeatureCSVBBox"] +"_Cam%s.csv"%Cam)
            KeyPoint2DPath = os.path.join(directoryName,settingsDict["FinalFeatureCSV2D"] +"_Cam%s.csv"%Cam)
            VideoPath = videoObjects[i].videoFilePath

            BBoxDF = pd.read_csv(BBoxPath)
            Key2DDF = pd.read_csv(KeyPoint2DPath)
            Subjects = settingsDict["Subjects"]
            # import ipdb;ipdb.set_trace()
            # GenerateSubject2DKeypointsCam(BBoxDF, Key2DDF,VideoPath,Subjects[:5],settingsDict,Cam,imsize)
            GenerateSubject2DKeypointsCam(BBoxDF, Key2DDF,VideoPath,Subjects,settingsDict,Cam,imsize)

    else:
        for i in range(4):
            Cam = settingsDict["cameras"][i]
            BBoxPath = os.path.join(directoryName,settingsDict["FinalFeatureCSVBBox"] +"_%s.csv"%Cam)
            KeyPoint2DPath = os.path.join(directoryName,settingsDict["FinalFeatureCSV2D"] +"_%s.csv"%Cam)
            VideoPath = videoObjects[i].videoFilePath

            BBoxDF = pd.read_csv(BBoxPath)
            Key2DDF = pd.read_csv(KeyPoint2DPath)
            Subjects = settingsDict["Subjects"]

            # import ipdb; ipdb.set_trace()
            if len(Subjects)>5:
                print("Done with loop 1")
                GenerateSubject2DKeypointsCam(BBoxDF, Key2DDF,VideoPath,Subjects[0:4],settingsDict,Cam,imsize, fps=30)
                print("Done with loop 2")
                GenerateSubject2DKeypointsCam(BBoxDF, Key2DDF,VideoPath,Subjects[4:7],settingsDict,Cam,imsize, fps=30)
                print("Done with loop 3")
                GenerateSubject2DKeypointsCam(BBoxDF, Key2DDF,VideoPath,Subjects[7:10],settingsDict,Cam,imsize, fps=30)
                print("Done with Cam" + Cam)
            else:
                GenerateSubject2DKeypointsCam(BBoxDF, Key2DDF,VideoPath,Subjects,settingsDict,Cam,imsize, fps=30)


def ReadBBoxDict(FrameBBoxDict,Subjects):
    """Reads BBox dictionary subset extracted for each frame"""
    OutputDict = {}
    for subject in Subjects:

        OutputDict.update({subject:[list(FrameBBoxDict["%s_BBox_x"%subject].values())[0],
        list(FrameBBoxDict["%s_BBox_y"%subject].values())[0],
        list(FrameBBoxDict["%s_BBox_h"%subject].values())[0],
        list(FrameBBoxDict["%s_BBox_w"%subject].values())[0]
        ]})
        

    return OutputDict

def createDatabase2D(customFeatures):
    """
    Create databse structure for storage of reprojected 2D features

    :return:
    """

    defaultDataSeries = {"frame": 0}
    for feature in customFeatures:
        defaultDataSeries[ str(feature) + "_x"] = 0
        defaultDataSeries[str(feature) + "_y"] = 0


    return defaultDataSeries

def GetObjectList(df):
    """Get list of all objects in a df"""
    dfCol = list(df.columns)
    dfCol.remove('frame')
    KeypointNameList = []

    for i in range(len(dfCol)):
        if (i+1) % 2 ==0:
            KeypointNameList.append(dfCol[i].strip("_y"))

    return KeypointNameList

def ReadKey2DDict(FrameKey2DDict,KeypointNameList):
    """Read 2D keypoint dictionary"""
    OutputDict = {}
    for keypoint in KeypointNameList:
        OutputDict.update({keypoint:[list(FrameKey2DDict["%s_x"%keypoint].values())[0],
        list(FrameKey2DDict["%s_y"%keypoint].values())[0]
        ]})

    return OutputDict
    


def GenerateSubject2DKeypointsCamSlow(BBoxDF, Key2DDF,VideoPath,Subjects,settingsDict,Cam,imsize):
    """slow version, for vicon videos, memory problem"""
    # import ipdb; ipdb.set_trace()


    OutDictList =  []
    if settingsDict["Mode"] == "custom":
        VidOutDir = os.path.join(settingsDict["TrialDirectory"],"Subject2DKeypointVideos")
        DataOutDir =os.path.join(settingsDict["TrialDirectory"],"Subject2DKeypointData") 
    else:
        VidOutDir = os.path.join(settingsDict["rootDirectory"],"Subject2DKeypointVideos")
        DataOutDir =os.path.join(settingsDict["rootDirectory"],"Subject2DKeypointData")    

    VideoWriterList = []
        #prepare output dicts:
    for j in range(len(Subjects)):
        OutDictList.append({})
        VideoWriterList.append(cv.VideoWriter(os.path.join(VidOutDir,"%s_Cam%s_%s.mp4"%(settingsDict["session"],Cam,Subjects[j])), cv.VideoWriter_fourcc(*'mp4v'), 30, imsize))

        
    
    # for j in range(len(Subjects)):
    #     cv.namedWindow("Window%i"%j, cv.WINDOW_NORMAL)
    for k in range(len(Subjects)):
        cap = cv.VideoCapture(VideoPath)
        cv.namedWindow("Window", cv.WINDOW_NORMAL)

        counter = BBoxDF["frame"][0]
        VidFrameCounter = 0#frame counter for specific video
        cap.set(cv.CAP_PROP_POS_FRAMES,counter) 

        # import ipdb;ipdb.set_trace()
        # imsize = (320,320) #set to this for now

        # BBoxDF.hist(column="391_3006_BBox_h")
        # plt.show()

        KeypointNameList = GetObjectList(Key2DDF)
        LastFrame = BBoxDF["frame"][len(BBoxDF)-1]

        # import ipdb;ipdb.set_trace()
        while(cap.isOpened()):
            # print(counter)
            if counter == LastFrame:
                break
            ret,frame = cap.read()
            BBoxDFSub = BBoxDF.loc[BBoxDF["frame"]==counter]
            if len(BBoxDFSub) == 0:
                break
            FrameBBoxDict = BBoxDFSub.to_dict()
            FrameKey2DDict = Key2DDF.loc[Key2DDF["frame"]==counter].to_dict()

            BBoxDict = ReadBBoxDict(FrameBBoxDict,Subjects)
            Key2DDict = ReadKey2DDict(FrameKey2DDict,KeypointNameList)

            if ret == True:
                # for i in range(len(Subjects)):
                SubBBox = BBoxDict[Subjects[k]]

                if np.isnan(SubBBox).any() or SubBBox[2]>imsize[1] or SubBBox[3]>imsize[0]:
                    #No bbox, or bbox not valid, draw black
                    PadCrop = np.zeros((imsize[0], imsize[1],3), dtype=np.uint8)
                    FrameDictKeyPoint = {}
                    for key,val in Key2DDict.items():
                        if Subjects[k] in key:
                            FrameDictKeyPoint.update({key:[math.nan,math.nan]})
                        else:
                            continue

                else:
                    SubBBox = [round(x) for x in SubBBox]
                    ##SubBBox: x, y, h, w

                    Crop = frame[SubBBox[1]:SubBBox[1]+SubBBox[2], SubBBox[0]:SubBBox[0]+SubBBox[3]]
                    # cv.waitKey(0)
                    if Crop.shape[0] %2 ==0:
                        #Shape is even number
                        YPadTop = int((imsize[1] - Crop.shape[0])/2)
                        YPadBot = int((imsize[1] - Crop.shape[0])/2)
                    else:
                        YPadTop = int( ((imsize[1] - Crop.shape[0])/2)-0.5)
                        YPadBot = int(((imsize[1] - Crop.shape[0])/2)+0.5)
                    ##Padding:
                    if Crop.shape[1] %2 ==0:
                        #Shape is even number
                        XPadLeft = int((imsize[0] - Crop.shape[1])/2)
                        XPadRight= int((imsize[0] - Crop.shape[1])/2)
                    else:
                        XPadLeft =  int(((imsize[0] - Crop.shape[1])/2)-0.5)
                        XPadRight= int(((imsize[0] - Crop.shape[1])/2)+0.5)


                    PadCrop = cv.copyMakeBorder(Crop, YPadTop,YPadBot,XPadLeft,XPadRight,cv.BORDER_REPLICATE)
                    cv.imshow("Window", PadCrop)
                    cv.waitKey(1)

                    ##Save 2D keypoint info
                    FrameDictKeyPoint = {}
                    for key,val in Key2DDict.items():
                        if Subjects[k] in key:
                            FrameDictKeyPoint.update({key:[val[0]-SubBBox[0]+XPadLeft, val[1]-SubBBox[1]+YPadTop]})
                        else:
                            continue
                VideoWriterList[k].write(PadCrop)
                # cv.imshow(windows[i], PadCrop)
                # cv.waitKey(1)
                FinalDict = createDatabase2D(KeypointNameList)
                for feature in FrameDictKeyPoint:
                    point2D = FrameDictKeyPoint[feature]
                    FinalDict[feature + "_x"] = point2D[0]
                    FinalDict[feature+ "_y"] = point2D[1]
                FinalDict["OriginalFrame"] = counter
                FinalDict["frame"] = VidFrameCounter
                OutDictList[k].update({str(counter):FinalDict})

                counter += 1
                VidFrameCounter += 1
            else:
                break

        cap.release()
        cv.destroyAllWindows()
        ##Save all to csv
        # for i in range(len(Subjects)):
        VideoWriterList[k].release()
        OutDir = os.path.join(DataOutDir,"%s_Cam%s_%s.csv"%(settingsDict["session"],Cam,Subjects[k]))
        outDF = pd.DataFrame.from_dict(OutDictList[k], orient="index")
        outDF.to_csv(OutDir, index=False)


def GenerateSubject2DKeypointsCam(BBoxDF, Key2DDF,VideoPath,Subjects,settingsDict,Cam,imsize,fps = 30):
    OutDictList =  []
    
    # cv.namedWindow("Window1", cv.WINDOW_NORMAL)
    # cv.namedWindow("Window2", cv.WINDOW_NORMAL)
    # windows = ["Window1","Window2"]
    cap = cv.VideoCapture(VideoPath)
    counter = BBoxDF["frame"][0]
    VidFrameCounter = 0#frame counter for specific video
    cap.set(cv.CAP_PROP_POS_FRAMES,counter) 

    # imsize = (320,320) #set to this for now
    # VidOutDir = os.path.join(settingsDict["TrialDirectory"],"Subject2DKeypointVideos")
    # DataOutDir =os.path.join(settingsDict["TrialDirectory"],"Subject2DKeypointData") 

    if settingsDict["Mode"] == "custom":
        VidOutDir = os.path.join(settingsDict["TrialDirectory"],"Subject2DKeypointVideos")
        DataOutDir =os.path.join(settingsDict["TrialDirectory"],"Subject2DKeypointData") 
    else:
        VidOutDir = os.path.join(settingsDict["rootDirectory"],"Subject2DKeypointVideos")
        DataOutDir =os.path.join(settingsDict["rootDirectory"],"Subject2DKeypointData")    


    # BBoxDF.hist(column="391_3006_BBox_h")
    # plt.show()
    VideoWriterList = []
    #prepare output dicts:
    for i in range(len(Subjects)):
        OutDictList.append({})
        VideoWriterList.append(cv.VideoWriter(os.path.join(VidOutDir,"%s_Cam%s_%s.mp4"%(settingsDict["session"],Cam,Subjects[i])), cv.VideoWriter_fourcc(*'mp4v'), fps, imsize))

    
    KeypointNameList = GetObjectList(Key2DDF)
    LastFrame = BBoxDF["frame"][len(BBoxDF)-1]


    while(cap.isOpened()):
        if counter == LastFrame:
            break
        ret,frame = cap.read()
        BBoxDFSub = BBoxDF.loc[BBoxDF["frame"]==counter]
        if len(BBoxDFSub) == 0:
            break
        FrameBBoxDict = BBoxDFSub.to_dict()
        FrameKey2DDict = Key2DDF.loc[Key2DDF["frame"]==counter].to_dict()

        BBoxDict = ReadBBoxDict(FrameBBoxDict,Subjects)
        Key2DDict = ReadKey2DDict(FrameKey2DDict,KeypointNameList)

        if ret == True:
            for i in range(len(Subjects)):
                SubBBox = BBoxDict[Subjects[i]]

                if np.isnan(SubBBox).any() or SubBBox[2]>imsize[1] or SubBBox[3]>imsize[0]:
                    #No bbox, or bbox not valid, draw black
                    PadCrop = np.zeros((imsize[0], imsize[1],3), dtype=np.uint8)
                    FrameDictKeyPoint = {}
                    for key,val in Key2DDict.items():
                        if Subjects[i] in key:
                            FrameDictKeyPoint.update({key:[math.nan,math.nan]})
                        else:
                            continue

                else:
                    SubBBox = [round(x) for x in SubBBox]
                    ##SubBBox: x, y, h, w

                    Crop = frame[SubBBox[1]:SubBBox[1]+SubBBox[2], SubBBox[0]:SubBBox[0]+SubBBox[3]]
                    # cv.imshow("Window", Crop)
                    # cv.waitKey(0)
                    if Crop.shape[0] %2 ==0:
                        #Shape is even number
                        YPadTop = int((imsize[1] - Crop.shape[0])/2)
                        YPadBot = int((imsize[1] - Crop.shape[0])/2)
                    else:
                        YPadTop = int( ((imsize[1] - Crop.shape[0])/2)-0.5)
                        YPadBot = int(((imsize[1] - Crop.shape[0])/2)+0.5)
                    ##Padding:
                    if Crop.shape[1] %2 ==0:
                        #Shape is even number
                        XPadLeft = int((imsize[0] - Crop.shape[1])/2)
                        XPadRight= int((imsize[0] - Crop.shape[1])/2)
                    else:
                        XPadLeft =  int(((imsize[0] - Crop.shape[1])/2)-0.5)
                        XPadRight= int(((imsize[0] - Crop.shape[1])/2)+0.5)


                    PadCrop = cv.copyMakeBorder(Crop, YPadTop,YPadBot,XPadLeft,XPadRight,cv.BORDER_REPLICATE)
                    ##Save 2D keypoint info
                    FrameDictKeyPoint = {}
                    for key,val in Key2DDict.items():
                        if Subjects[i] in key:
                            FrameDictKeyPoint.update({key:[val[0]-SubBBox[0]+XPadLeft, val[1]-SubBBox[1]+YPadTop]})
                        else:
                            continue
                VideoWriterList[i].write(PadCrop)
                # cv.imshow(windows[i], PadCrop)
                # cv.waitKey(1)
                FinalDict = createDatabase2D(KeypointNameList)
                for feature in FrameDictKeyPoint:
                    point2D = FrameDictKeyPoint[feature]
                    FinalDict[feature + "_x"] = point2D[0]
                    FinalDict[feature+ "_y"] = point2D[1]
                FinalDict["OriginalFrame"] = counter
                FinalDict["frame"] = VidFrameCounter
                OutDictList[i].update({str(counter):FinalDict})

            counter += 1
            VidFrameCounter += 1
        else:
            break


    cap.release()
    cv.destroyAllWindows()
    ##Save all to csv
    for i in range(len(Subjects)):
        VideoWriterList[i].release()
        OutDir = os.path.join(DataOutDir,"%s_Cam%s_%s.csv"%(settingsDict["session"],Cam,Subjects[i]))
        outDF = pd.DataFrame.from_dict(OutDictList[i], orient="index")
        outDF.to_csv(OutDir, index=False)
