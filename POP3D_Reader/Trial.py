# !/usr/bin/env python3
""" Defines the trial class, with all data loaded"""
import sys
sys.path.append("./")

import os
import csv
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt


from POP3D_Reader import CameraObject
from Util.VisualizeUtil import *


class Trial:
    
    def __init__(self, dataDir,SequenceNum):
        """
        Initialize Trial Class, loads everything related to given trial
        
        Input:
        dataDir [str]: Directory to dataset
        SequenceNum [int]: Sequence number
   
               
        """
        # import ipdb;ipdb.set_trace()
        
        #Read Metadata:
        MetaData = pd.read_csv(os.path.join(dataDir,"Pop3DMetadata.csv"), dtype = str)
        RowDict = MetaData.loc[MetaData["Sequence"] == str(SequenceNum)].to_dict()
        self.TrialName = "Sequence%s_n%02d_%s"%(list(RowDict["Sequence"].values())[0],int(list(RowDict["IndividualNum"].values())[0]),list(RowDict["Date"].values())[0])
        self.baseDir = os.path.join(dataDir,"Pigeon%02d"%int(list(RowDict["IndividualNum"].values())[0]),self.TrialName)
        
        self.FileNameDict = self.GenerateFileNames()
        self.CamRes = (3840,2160)
        self.camObjects = self.loadCamObjects()
        self.CamNum = 4
        self.Subjects = list(RowDict["Subjects"].values())[0].split(";")
        # import ipdb; ipdb.set_trace()

    def GenerateFileNames(self, Type = "3DPOP"):
        """
        From row from meta data, generate all required file names into a dictionary
        Type: type of data, can be: [3DPOP, Train, Val, Test]        
        """
        
        if Type == "3DPOP":
            FileNameDict = {"VideoPaths": [os.path.join(self.baseDir, "Videos","%s-Cam%s.mp4"%(self.TrialName, i+1) ) for i in range(4)],
                            "ExtrinsicPaths":[os.path.join(self.baseDir, "CalibrationInfo","%s-Cam%s-Extrinsics.p"%(self.TrialName, i+1) ) for i in range(4)],
                            "IntrinsicPaths":[os.path.join(self.baseDir, "CalibrationInfo","%s-Cam%s-Intrinsics.p"%(self.TrialName, i+1) ) for i in range(4)],
                            "BBoxPaths":[os.path.join(self.baseDir, "Annotation","%s-Cam%s-BBox.csv"%(self.TrialName, i+1) ) for i in range(4)],
                            "Key2DPaths":[os.path.join(self.baseDir, "Annotation","%s-Cam%s-Keypoint2D.csv"%(self.TrialName, i+1) ) for i in range(4)],
                            "Key2DFilterPaths":[os.path.join(self.baseDir, "Annotation","%s-Cam%s-Keypoint2DFiltered.csv"%(self.TrialName, i+1) ) for i in range(4)],
                            "Key3DPaths":[os.path.join(self.baseDir, "Annotation","%s-Cam%s-Keypoint3D.csv"%(self.TrialName, i+1) ) for i in range(4)],
                            "SyncArrPaths":[os.path.join(self.baseDir, "CalibrationInfo","%s-Cam%s-SyncArray.p"%(self.TrialName, i+1) ) for i in range(4)]}
        elif Type in ["Train", "Val", "Test"]:
                        FileNameDict = {"VideoPaths": [os.path.join(self.baseDir, "TrainingSplit",Type,"%s-Cam%s.mp4"%(self.TrialName, i+1) ) for i in range(4)],
                            "ExtrinsicPaths":[os.path.join(self.baseDir, "CalibrationInfo","%s-Cam%s-Extrinsics.p"%(self.TrialName, i+1) ) for i in range(4)],
                            "IntrinsicPaths":[os.path.join(self.baseDir, "CalibrationInfo","%s-Cam%s-Intrinsics.p"%(self.TrialName, i+1) ) for i in range(4)],
                            "BBoxPaths":[os.path.join(self.baseDir,"TrainingSplit",Type,"%s-Cam%s-BBox.csv"%(self.TrialName, i+1) ) for i in range(4)],
                            "Key2DPaths":[os.path.join(self.baseDir, "TrainingSplit",Type,"%s-Cam%s-Keypoint2D.csv"%(self.TrialName, i+1) ) for i in range(4)],
                            "Key2DFilterPaths":[os.path.join(self.baseDir, "TrainingSplit",Type,"%s-Cam%s-Keypoint2DFiltered.csv"%(self.TrialName, i+1) ) for i in range(4)],
                            "Key3DPaths":[os.path.join(self.baseDir, "TrainingSplit",Type,"%s-Cam%s-Keypoint3D.csv"%(self.TrialName, i+1) ) for i in range(4)],
                            "SyncArrPaths":[os.path.join(self.baseDir, "CalibrationInfo","%s-Cam%s-SyncArray.p"%(self.TrialName, i+1) ) for i in range(4)]}
        
        return(FileNameDict)

    def loadCamObjects(self):
        """Load Camera objects"""
        CamObj = []
        CamNames = ["Cam%i"%(i+1) for i in range(4)]
        
        for index in range(4):
            CamName = CamNames[index]
            VidPath = self.FileNameDict["VideoPaths"][index]
            ExtPath = self.FileNameDict["ExtrinsicPaths"][index]
            IntPath = self.FileNameDict["IntrinsicPaths"][index]
            CamObject = CameraObject.CameraObject(CamName,VidPath,ExtPath,IntPath,self.CamRes)            
            CamObj.append(CamObject)
        return CamObj
    
    def load3DPopDataset(self, Filter = False):
        """
        Load in all ground truth data from 3D POP
        
        Filter: Whether to load filtered csv for 2D keypoints
        """
        
        for index in range(self.CamNum):
            ##Paths:
            Keypoint2DPath = self.FileNameDict["Key2DPaths"][index]
            BBoxPath = self.FileNameDict["BBoxPaths"][index]
            Keypoint3DPath = self.FileNameDict["Key3DPaths"][index]
            FilterKeypoint2DPath = self.FileNameDict["Key2DFilterPaths"][index]

            self.camObjects[index].loadBBoxData(BBoxPath)
            self.camObjects[index].load3DKeypoint(Keypoint3DPath)

            if Filter:
                self.camObjects[index].Keypoint2D = self.camObjects[index].ReadCSV(FilterKeypoint2DPath)
            else:
                self.camObjects[index].load2DKeypoint(Keypoint2DPath)
                self.camObjects[index].load2DFilterKeypoint(FilterKeypoint2DPath)
                
    def load3DPopTrainingSet(self, Filter = True, Type = "Train"):
        """
        Load in training set for 3D POP
        
        Filter: Whether to load filtered csv for 2D keypoints
        Type: type of data, can be: [Train, Val, Test]  
        """
        self.FileNameDict = self.GenerateFileNames(Type=Type)
        self.camObjects = self.loadCamObjects()
        
        for index in range(self.CamNum):
            ##Paths:
            Keypoint2DPath = self.FileNameDict["Key2DPaths"][index]
            BBoxPath = self.FileNameDict["BBoxPaths"][index]
            Keypoint3DPath = self.FileNameDict["Key3DPaths"][index]
            FilterKeypoint2DPath = self.FileNameDict["Key2DFilterPaths"][index]

            self.camObjects[index].loadBBoxData(BBoxPath)
            self.camObjects[index].load3DKeypoint(Keypoint3DPath)

            if Filter:
                self.camObjects[index].Keypoint2D = self.camObjects[index].ReadCSV(FilterKeypoint2DPath)
            else:
                self.camObjects[index].load2DKeypoint(Keypoint2DPath)
                self.camObjects[index].load2DFilterKeypoint(FilterKeypoint2DPath)

    def VisualizeTrainingData(self,CamIndex = 0, startframe = 0,
                              save = False, 
                              show= True,
                              points = True, 
                              Lines= False, 
                              BBox=False,
                              Traj=False,
                              MarkersOnly=False,
                              jupyter = False):
        
        """
        Master Function to allow visualization of all keypoints
        
        CamIndex: Index of the camera within the trial object
        starframe: which frame to start visualization
        
        Options [Bool]:
        save: Saves video to specified directory
        show: Shows video on a opencv window
        points: plots all keypoints as points
        Lines: plots lines between the keypoints to visualize head and body objects
        BBox: plots bounding boxes
        Traj: plots trajectories
        MarkersOnly = plots motion tracking markers only
        
        """
        counter=startframe
        camObj = self.camObjects[CamIndex]
        
        ##Load in required data:
        VideoPath = camObj.VideoPath
        
        BirdList = self.Subjects
        # BirdList = ["485_0107"]
        #custom colours for trajectory:
        TrajColours = [(255, 0 , 0 ),(0,255,0), (0,0,255),(255,255,0),(255, 0 , 255),(0, 255, 255),(63,133,205),(128,0,128),(203,192,255),(0,165,255)]

        if show:
            cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
        cap = cv2.VideoCapture(VideoPath)
        cap.set(cv2.CAP_PROP_POS_FRAMES,counter) #Start reading frames from startframe

        # record vid
        if save:
            out = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*'mp4v'), 30, camObj.dim)

        BirdTrajDict = {}
        for bird in BirdList:
            BirdTrajDict[bird] = []

        while(cap.isOpened()):
            if counter > max(camObj.Keypoint2D["frame"]): #end if reach end of dataframe
                break
        # for x in range(1800):
            # print(counter)
            ret, frame = cap.read()
            if ret == True:
                for k in range(len(BirdList)):
                    bird = BirdList[k]
                    MarkerDict = camObj.Read2DKeypointData(camObj.Keypoint2D, counter, bird)
                    # AllMarkers, AllObjNames = GetMarkerPoints2D(camObj.Keypoint2D, bird, counter)
                    if len(MarkerDict)==0:
                        continue
                        
                    if Lines:
                        #Draw lines
                        PlotLine(MarkerDict,"leftEye","nose",[0,0,255],frame)
                        PlotLine(MarkerDict,"rightEye","nose",[0,0,255],frame)
                        PlotLine(MarkerDict,"beak","nose",[0,0,255],frame)
                        PlotLine(MarkerDict,"leftEye","rightEye",[0,0,255],frame)
                        PlotLine(MarkerDict,"leftShoulder","rightShoulder",[0,255,0],frame)
                        PlotLine(MarkerDict,"leftShoulder","topKeel",[0,255,0],frame)
                        PlotLine(MarkerDict,"topKeel","rightShoulder",[0,255,0],frame)
                        PlotLine(MarkerDict,"leftShoulder","tail",[0,255,0],frame)
                        PlotLine(MarkerDict,"tail","rightShoulder",[0,255,0],frame)
                        PlotLine(MarkerDict,"tail","bottomKeel",[0,255,0],frame)
                        PlotLine(MarkerDict,"bottomKeel","topKeel",[0,255,0],frame)
                    
                    if Traj:
                        Start, End = camObj.GetBBoxData(camObj.BBox, counter,bird)
                        if not Start or not End:
                            continue
                        elif np.isnan([Start[0],Start[1],End[0],End[1]]).any() :
                            continue
                        Midpoint = camObj.GetBBoxMidPoint(camObj.BBox, counter,bird)
                        BirdTrajDict[bird].append(Midpoint)
                        # import ipdb;ipdb.set_trace()
                        PlotBirdTrajectory(frame, BirdTrajDict[bird], 150, TrajColours[k])

                    if BBox:
                        Start, End = camObj.GetBBoxData(camObj.BBox, counter,bird)
                        if not Start or not End:
                            continue
                        elif np.isnan([Start[0],Start[1],End[0],End[1]]).any() :
                            continue

                        # print(Start)
                        cv2.rectangle(frame,(round(Start[0]),round(Start[1])),(round(End[0]),round(End[1])),
                        TrajColours[k],3)
                    if points:
                        for key,coord in MarkerDict.items():
                            pts = coord
                            if np.isnan(pts[0]) or math.isinf(pts[0]) or math.isinf(pts[1]):
                                continue
                            #######
                            point = (round(pts[0]),round(pts[1]))
                            if IsPointValid(camObj.dim,point):
                                # import ipdb;ipdb.set_trace()
                                if MarkersOnly:
                                    Type = getType(key)
                                    if Type is None:
                                        colour = getColor(key)
                                        cv2.circle(frame,point,3,colour, -1)
                                    else:
                                        continue

                                if Lines:
                                    Type = getType(key)
                                    if Type == "Head":
                                        cv2.circle(frame,point,2,[0,255,255], -1)
                                    elif Type =="Backpack":
                                        cv2.circle(frame,point,2,[0,255,255], -1)
                                    else:
                                        continue
                                else:
                                    colour = getColor(key)
                                    cv2.circle(frame,point,1,colour, -1)

                if save:
                    out.write(frame)
                if show:
                    cv2.imshow('Window',frame)
                    # cv2.imwrite('./sample.jpg', frame)

                if jupyter:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    plt.imshow(frame)
                    plt.show()
                    break
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
            else:
                break
            counter += 1
            
        cap.release()
        cv2.destroyAllWindows()
        if save:  
            out.release()

def main():
    ##Test on default file
    TrialPath = "/media/alexchan/My Passport/Pop3D-Dataset_Final/"
    Sequence = 24
    PigeonTrial = Trial(TrialPath,Sequence)
    PigeonTrial.load3DPopDataset(Filter=False)
    # PigeonTrial.load3DPopTrainingSet(Filter=False, Type="Val")
    # import ipdb;ipdb.set_trace()
    
    PigeonTrial.VisualizeTrainingData(CamIndex=0,startframe = 1000,
                                    save = False, 
                                    show=True,
                                    points = True, 
                                    Lines= True, 
                                    BBox=True,
                                    Traj=True,
                                    MarkersOnly=False)
    
if __name__ == "__main__":
    main()