"""
Author: Alex Chan

Part of the auto-keypoint annotation pipeline, takes 2D keypoint from each view,
outputs CSV for BBox for each subject

"""

import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import math


def computeSubjectBoundingBox(Dict,subject,offset = 40):
    """
    Author: Alex Chan
    Compute the corners required for drawing the bounding box for single subject
    Looks for keypoint with the most extreme values
    Note: output is little different from parent function, corners output is always
    top left corner and bottom right corner
    :return:
    """
    # import ipdb; ipdb.set_trace()

    bBox = {}
    ###compute leftmost, rightmost, topmost and bottommost point
    leftmost = float("Inf")
    rightmost = -float("Inf")
    topmost = float("Inf")
    bottommost = -float("Inf")

    SubjectDict = {k:v for k,v in Dict.items() if k.startswith(subject)}
    

    ##If head or backpack lose tracking, return NA, down stream will use bbox from previous frame
    TestNA = [v for k,v in SubjectDict.items() if k.endswith(("hd1","hd2","hd3","hd4","bp1","bp2","bp3","bp4"))]
    if np.isnan(TestNA).any():
        bBox["lCorner"] = [math.nan,math.nan]
        bBox["rCorner"] = [math.nan,math.nan]
        return bBox

    for point in SubjectDict.values():
        if point == [0,0] or point == [math.nan,math.nan]:
            continue
        else:#Update points if the point is larger/smaller than most extreme point so far
            if point[0] < leftmost:
                leftmost = point[0]
            if point[0] > rightmost:
                rightmost = point[0]
            if point[1] < topmost:
                topmost= point[1]
            if point[1] > bottommost:
                bottommost = point[1]

    # import ipdb; ipdb.set_trace()
    if any([math.isinf(x) for x in [leftmost,rightmost,topmost,bottommost]]):
        ##There is still infinite, just return NA
        bBox["lCorner"] = [math.nan,math.nan]
        bBox["rCorner"] = [math.nan,math.nan]
        return bBox
        
    lCorner = [leftmost - offset, topmost - offset + round(offset/3) ]
    rCorner = [rightmost + offset, bottommost + offset]
        
    lCorner = [0 if point <0 else point for point in lCorner]
    rCorner = [0 if point <0 else point for point in rCorner]

    bBox["lCorner"] = lCorner
    bBox["rCorner"] = rCorner

    return bBox

def GenerateBBoxCam(df, Subjects):
    """Generates BBox csv for each camera"""
    FinalDict = {}

    PrevFrameSubjectDict = {}
    for subject in Subjects:        
        SubjectDict = {"%s_BBox_x"%subject: math.nan,
                "%s_BBox_y"%subject: math.nan,
                "%s_BBox_w"%subject: math.nan,
                "%s_BBox_h"%subject: math.nan
                }
        PrevFrameSubjectDict[subject] = SubjectDict

    for i,frame in enumerate(tqdm(df["frame"])):

        FrameData = df.loc[i].to_dict()
        FrameData.pop("frame")

        #read data
        counter = 0
        Point2D =[None]*2
        KeypointNameList = []
        Point2DList=[]
        
        for key,val in FrameData.items():
            Point2D[counter] = val
            counter +=1

            if counter ==2:
                Point2DList.append(Point2D)
                Point2D = [None]*2
                counter = 0
                KeypointNameList.append(key.strip("_y"))

        FrameDict = {KeypointNameList[x]:Point2DList[x] for x in range(len(Point2DList))}
        FrameBBoxDict = {}
        

        for subject in Subjects:
            BBoxDict = computeSubjectBoundingBox(FrameDict, subject, offset = 60)
            # import ipdb; ipdb.set_trace()

            if np.isnan(list(BBoxDict.values())).any():
                #if Na, use previous bounding box
                SubjectDict = PrevFrameSubjectDict[subject]
            else:
                SubjectDict = {"%s_BBox_x"%subject: BBoxDict["lCorner"][0],
                "%s_BBox_y"%subject: BBoxDict["lCorner"][1],
                "%s_BBox_w"%subject: BBoxDict["rCorner"][0]-BBoxDict["lCorner"][0],
                "%s_BBox_h"%subject: BBoxDict["rCorner"][1]-BBoxDict["lCorner"][1]
                }
            PrevFrameSubjectDict[subject] = SubjectDict
            FrameBBoxDict.update(SubjectDict)

        FrameBBoxDict["frame"] = frame
        FinalDict.update({str(frame):FrameBBoxDict})

    return FinalDict



def GenerateBBox(projectSettings):
    """Generate bounding box for each subject"""
    settingsDict = projectSettings.settingsDict
    AnnotDir = settingsDict["AnnotationDirectory"]
    Subjects = settingsDict["Subjects"]

    #loop through cameras:
    for i in range(4):
        Cam = i+1
        path = os.path.join(AnnotDir,settingsDict["FinalFeatureCSV2D"][i])
        df = pd.read_csv(path)
        OutDict = GenerateBBoxCam(df, Subjects)
        OutDF = pd.DataFrame.from_dict(OutDict, orient="index")
        OutPath = os.path.join(AnnotDir,settingsDict["FinalFeatureCSVBBox"][i])
        OutDF.to_csv(OutPath, index=False)
   


    
