# !/usr/bin/env python3
"""Sample images and save annotaitons to json to be read by pytorch dataloader"""

import sys
sys.path.append("./")
from POP3D_Reader import Trial
import os
import cv2
import numpy as np
import math
import pandas as pd
import random
from tqdm import tqdm
import json
random.seed(10)


def GetInstancePerTrial(MetaDataDir,DatasetDir,TrainNum,ValRatio,TestRatio):
    """
    Given number of training images, determine how many images to sample for each trial
    * Assume equal number of images for each individual num (1,2,5,10)
    """
    Metadf = pd.read_csv(MetaDataDir,dtype=str)
    #Types of individual number:
    IndividualNumType = sorted(list(set(Metadf["IndividualNum"].to_list())))
    NumPerType = math.ceil(TrainNum/len(IndividualNumType))
    print("Number of images per individual number types: %i" %(NumPerType))
    TotalNumToSampleDict = {"Train":NumPerType,"Val":NumPerType*ValRatio,"Test":NumPerType*TestRatio}
    
    
    ##Prepare out dicts
    TrainImgtoSampleDict = {}
    ValImgtoSampleDict = {}
    TestImgtoSampleDict = {}
    ImgtoSampleDicts = {"Train":TrainImgtoSampleDict,"Val":ValImgtoSampleDict,"Test":TestImgtoSampleDict}
    
    ##ImageSampled:
    TrainImgSampled = {'1':0,'2':0,'5':0,'10':0}
    ValImgSampled ={'1':0,'2':0,'5':0,'10':0}
    TestImgSampled = {'1':0,'2':0,'5':0,'10':0}
    ImgSampledDict = {"Train":TrainImgSampled,"Val":ValImgSampled,"Test":TestImgSampled}
    
    Types = ["Train", "Val","Test"]
    
    for IndNum in IndividualNumType:
        #Subset df for this ind number
        IndDF = Metadf.loc[Metadf["IndividualNum"] == IndNum]
        ImgtoSampleTrain = math.ceil(NumPerType/len(IndDF))
        ImgtoSampleVal = math.ceil(ImgtoSampleTrain*ValRatio)
        ImgtoSampleTest = math.ceil(ImgtoSampleTrain*TestRatio)
        
        #Number of images to sample for each type:
        TypeNums = {"Train":ImgtoSampleTrain,"Val":ImgtoSampleVal,"Test":ImgtoSampleTest}

        TotalNumCounter = {"Train":0, "Val":0, "Test":0} #Counter to keep track of total frames sampled for certain type
        print("Calculating %s Individuals:"%IndNum)
        for index, row in tqdm(IndDF.iterrows()):
            PigeonTrial = Trial.Trial(DatasetDir,row["Sequence"])
            for Type in Types:            
                PigeonTrial.load3DPopTrainingSet(Filter = True, Type = Type)
                #Find frames where all camera views no NA
                FramesList = []
                for camObj in PigeonTrial.camObjects:
                    FramesList.append(camObj.Keypoint2D.dropna(axis=0)["frame"].to_list())
                NoNAFrameList = sorted(list(set(FramesList[0]) & set(FramesList[1]) & set(FramesList[2])& set(FramesList[3])))
                
                # if IndNum == "10":
                #     import ipdb;ipdb.set_trace()
                
                if IndNum == "10" and row["Sequence"] == "59": #if 10 just sample all
                    #Sample all:
                    # ImgtoSampleDicts[Type].update({int(row["Sequence"]):len(NoNAFrameList)})
                    # TotalNumCounter[Type] += len(NoNAFrameList)
                    # ImgSampledDict[Type][IndNum] += len(NoNAFrameList)
                    
                    ##Smaple till enough:
                    # import ipdb;ipdb.set_trace()
                    SampleValue = (TotalNumToSampleDict[Type]- TotalNumCounter[Type])
                    ImgtoSampleDicts[Type].update({int(row["Sequence"]):(SampleValue)})
                    TotalNumCounter[Type] +=SampleValue
                    ImgSampledDict[Type][IndNum] +=SampleValue
                    continue

                
                if len(NoNAFrameList) > TypeNums[Type]: #if have plenty frames to sample from
                    ImgtoSampleDicts[Type].update({int(row["Sequence"]):TypeNums[Type]})
                    TotalNumCounter[Type]+=TypeNums[Type]
                    ImgSampledDict[Type][IndNum] += TypeNums[Type]
                    
                elif TotalNumCounter[Type] + len(NoNAFrameList) > TotalNumToSampleDict[Type]: #if after this trial can have enough, dont sample all, just get enough
                    SampleValue = (TotalNumToSampleDict[Type]- TotalNumCounter[Type])
                    ImgtoSampleDicts[Type].update({int(row["Sequence"]):(SampleValue)})
                    TotalNumCounter[Type] +=SampleValue
                    ImgSampledDict[Type][IndNum] +=SampleValue

                else: #Else just sample all
                    ImgtoSampleDicts[Type].update({int(row["Sequence"]):len(NoNAFrameList)})
                    TotalNumCounter[Type] += len(NoNAFrameList)
                    ImgSampledDict[Type][IndNum] += len(NoNAFrameList)

    for Type in Types:                
        print(Type)  
        print("Approx Image to sample: %s"%(TotalNumToSampleDict[Type]))
        print("Total Images: %s"%ImgSampledDict[Type])
                    
                    
    # import ipdb;ipdb.set_trace()
    return ImgtoSampleDicts
    
def SaveImages(PigeonTrial, RandomFrames,OutDir,Keypoints,Type, MasterIndexCounter,DictList2D,DictList3D):
    """For a trial, save frames into output directory and return annotation dict list for 2D and 3D"""
    CamObjList = []
    CapList = []
    SaveDirList = []

    for camObj in PigeonTrial.camObjects:
        CamObjList.append(camObj)
        CapList.append(cv2.VideoCapture(camObj.VideoPath))
        SaveDirList.append(os.path.join(OutDir,camObj.CamName))
        
        
    SeqName = PigeonTrial.TrialName
    counter = 0

    while True:
        # print(counter)
        FrameList = [cap.read() for cap in CapList]
        # print(ret)
                
        if FrameList[0][0] == True:
            if counter in RandomFrames: 
                ##Frame included in sampled frames:
                BBoxDataList = []
                Data2DList = []
                
                for x in range(len(CamObjList)):
                    cv2.imwrite(os.path.join(SaveDirList[x],"%s-F%s.jpg"%(SeqName,counter)),FrameList[x][1])

                    #BBox Data
                    BBoxData =  {ID:list(CamObjList[x].GetBBoxData(CamObjList[x].BBox, counter, ID)) for ID in PigeonTrial.Subjects} 
                    BBoxData = {ID:[val[0][0],val[0][1],val[1][0],val[1][1]] for ID,val in BBoxData.items()}
                    BBoxDataList.append(BBoxData)
                    
                    #2D Data
                    Data2D =  {ID:CamObjList[x].Read2DKeypointData(CamObjList[x].Keypoint2D, counter, ID,Keypoints,StripName=True) for ID in PigeonTrial.Subjects} 
                    Data2DList.append(Data2D)

                    
                ##3D Data
                Data3D =  {ID:CamObjList[x].Read3DKeypointData(CamObjList[x].Keypoint3D, counter, ID,Keypoints,StripName=True) for ID in PigeonTrial.Subjects} 
                CameraDictList = []
                for x in range(len(CamObjList)):
                    CameraDict = {}
                    CameraDict["CamName"] = CamObjList[x].CamName
                    CameraDict["Path"]=os.path.join(Type,CamObjList[x].CamName,"%s-F%s.jpg"%(SeqName,counter))
                    CameraDict["BBox"]=BBoxDataList[x]
                    CameraDict["Keypoint2D"]= Data2DList[x]
                    CameraDictList.append(CameraDict)
                
                ##Save 3D all data
                DictList3D.append({
                    "Image-ID" : MasterIndexCounter, 
                    "BirdID" : PigeonTrial.Subjects,
                    "Keypoint3D": Data3D,
                    "CameraData": CameraDictList
                })
                
                #2D Data, sample random view between all cameras
                RandomCamIndex = random.sample(list(range(len(CamObjList))),1)[0]
                SaveImgPath = os.path.join(OutDir,"MixedViews","%s-%s-F%s.jpg"%(CamObjList[RandomCamIndex].CamName, SeqName,counter))
                cv2.imwrite(SaveImgPath,FrameList[RandomCamIndex][1])

                DictList2D.append({
                    "Image-ID" : MasterIndexCounter, 
                    "BirdID" : PigeonTrial.Subjects,
                    "Path" : os.path.join(Type,"MixedViews","%s-%s-F%s.jpg"%(CamObjList[RandomCamIndex].CamName, SeqName,counter)),
                    "Keypoint3D": Data3D,
                    "Keypoint2D": Data2DList[RandomCamIndex],
                    "BBox":BBoxDataList[RandomCamIndex]
                })
                # import ipdb;ipdb.set_trace()

                MasterIndexCounter +=1

            counter +=1
            # print(counter)  
        elif counter == 0:
            import ipdb;ipdb.set_trace()
            print("weird, cant read first frame from video")
            continue
        else:
            #end of video, write video
            break
            
    Release = [cap.release() for cap in CapList]
    return DictList3D,DictList2D,MasterIndexCounter


#Temp arguments:
# DatasetDir = DatasetDir
# OutDir = TrainDir
# ImgDict = TrainImgtoSampleDict
# AnnotationDir = AnnotationDir
# Type = "Train"

def SampleImages(DatasetDir,OutDir,ImgDict, AnnotationDir, Keypoints,Type):
    """
    Sample images for a type (train/val/test) and save annotation as json
    Extracts both 3D and 2D ground truth
    """
    
    if not os.path.exists(os.path.join(OutDir,"Cam1")):
        os.mkdir(os.path.join(OutDir,"Cam1"))
        os.mkdir(os.path.join(OutDir,"Cam2"))
        os.mkdir(os.path.join(OutDir,"Cam3"))
        os.mkdir(os.path.join(OutDir,"Cam4"))
        os.mkdir(os.path.join(OutDir,"MixedViews"))
    else:
        print("Directories already exist! Ensure Folders are cleared!!")

    MasterIndexCounter = 0
    DictList2D = []
    DictList3D = []    
    
    for Seq, NumImg in tqdm(ImgDict.items()):
        if NumImg == 0:
            continue
        
        PigeonTrial = Trial.Trial(DatasetDir,Seq)
        PigeonTrial.load3DPopTrainingSet(Filter = True, Type = Type)
        
        #Find frames where all camera views no NA
        FramesList = []
        for camObj in PigeonTrial.camObjects:
            FramesList.append(camObj.Keypoint2D.dropna(axis=0)["frame"].to_list())
        # import ipdb;ipdb.set_trace()
        NoNAFrameList = sorted(list(set(FramesList[0]) & set(FramesList[1]) & set(FramesList[2])& set(FramesList[3])))
        
        # if len(NoNAFrameList) == 0:
        #     continue
        if (NumImg/len(NoNAFrameList))*100 > 100: #if not enough images, just get all images from that trial
            import ipdb;ipdb.set_trace()

        print(Seq)
        print("Sampling %s %% of frames present in sequence" %((NumImg/len(NoNAFrameList))*100))
        
        RandomFrames = sorted(random.sample(NoNAFrameList,int(NumImg)))
        
        DictList3D, DictList2D,MasterIndexCounter = SaveImages(PigeonTrial, RandomFrames,OutDir,Keypoints,Type, MasterIndexCounter,DictList2D,DictList3D)

    # import ipdb;ipdb.set_trace()
    OutputDict3D = {
        "info" : {
        "Description":"Sampled 3D ground truth Data from 3D-POP dataset",
        "Collated by": "Alex Chan",
        "Date":"06/02/2023",
        "Keypoints": Keypoints,
        "TotalImages": sum(list(ImgDict.values()))
    },
      "Annotations":DictList3D}
    
    with open(os.path.join(AnnotationDir,"%s-3D.json"%Type), "w") as outfile:
        json.dump(OutputDict3D, outfile, indent=4)
    
    
    OutputDict2D = {
        "info" : {
        "Description":"Sampled 2D ground truth Data from 3D-POP dataset",
        "Collated by": "Alex Chan",
        "Date":"06/02/2023",
        "Keypoints": Keypoints,
        "TotalImages": sum(list(ImgDict.values()))
    },
      "Annotations":DictList2D}
    
    
    with open(os.path.join(AnnotationDir,"%s-2D.json"%Type), "w") as outfile:
        json.dump(OutputDict2D, outfile,indent=4)
    
    
    
###Temp function to move all calibration info to another folder

def CopyCalibrationFiles(DatasetDir,OutputDir,MetaDataDir):
    DatasetDir = "/media/alexchan/My Passport/Pop3D-Dataset_Final/"
    OutputDir = "/media/alexchan/My Passport/Pop3D-Dataset_Final/ImageTrainingData/N5000/Calibration/"
    MetaDataDir = os.path.join(DatasetDir,"Pop3DMetadata.csv")
    Metadf = pd.read_csv(MetaDataDir,dtype=str)
    
    if not os.path.exists(OutputDir):
        os.mkdir(OutputDir)
    

    for Seq in Metadf["Sequence"].tolist():
        PigeonSeq = Trial.Trial(DatasetDir, int(Seq))

        FileNames = PigeonSeq.GenerateFileNames()
    
        FilesCopy = FileNames["IntrinsicPaths"] + FileNames["ExtrinsicPaths"]
    
        import shutil
        for file in FilesCopy:
            shutil.copy(file,OutputDir )


    

    
def main(DatasetDir,OutputDir,TrainNum,ValRatio,TestRatio,Keypoints):
    MetaDataDir = os.path.join(DatasetDir,"Pop3DMetadata.csv")
    ImgtoSampleDicts= GetInstancePerTrial(MetaDataDir,DatasetDir,TrainNum,ValRatio,TestRatio)

    TrainImgtoSampleDict = ImgtoSampleDicts["Train"]
    ValImgtoSampleDict = ImgtoSampleDicts["Val"]
    TestImgtoSampleDict = ImgtoSampleDicts["Test"]
    
    TrainDir = os.path.join(OutputDir,"Train")
    ValDir = os.path.join(OutputDir,"Val")
    TestDir = os.path.join(OutputDir,"Test")
    AnnotationDir = os.path.join(OutputDir, "Annotation")
    
    if not os.path.exists(TrainDir):
        os.mkdir(TrainDir)
        os.mkdir(ValDir)
        os.mkdir(TestDir)
        os.mkdir(AnnotationDir)
        
    # SampleImages(DatasetDir,TrainDir,TrainImgtoSampleDict,AnnotationDir,Keypoints,Type = "Train")
    # SampleImages(DatasetDir,ValDir,ValImgtoSampleDict,AnnotationDir,Keypoints,Type = "Val")
    SampleImages(DatasetDir,TestDir,TestImgtoSampleDict,AnnotationDir,Keypoints,Type = "Test")

    

if __name__ == "__main__":
    DatasetDir = "/media/alexchan/My Passport/Pop3D-Dataset_Final/"
    OutputDir = "/home/alexchan/Documents/SampleDatasets/ImageTrainingData/N100"
    
    Keypoints = ["hd_beak","hd_leftEye","hd_rightEye","hd_nose","bp_leftShoulder","bp_rightShoulder","bp_topKeel","bp_bottomKeel","bp_tail"]
    
    TrainNum = 100 #Number of training images
    ValRatio = 0.2 #ratio for validation and test
    TestRatio = 0.1 

    main(DatasetDir,OutputDir,TrainNum,ValRatio,TestRatio,Keypoints)
