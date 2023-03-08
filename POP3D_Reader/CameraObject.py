# !/usr/bin/env python3
""" Defines Sony Camera Objects"""

import os
import csv
import pickle as p
import numpy as np
import pandas as pd
import cv2


class CameraObject:
    
    def __init__(self,CamName, VideoPath, ExtPath,IntPath,Resolution):
        """
        Initialize Camera Object
        CamName [str]: Name of Camera
        VideoPath [str]: Path to video
        ExtPath [str]: Path to extrinsics pickle
        IntPath [str]: Path to intrinsics pickle
        Resolution [tuple]: (width,height) of input video        
        """
        
        # import ipdb;ipdb.set_trace()
        self.CamName = CamName
        self.VideoPath = VideoPath
        self.dim = Resolution
        
        ##Load camera Extrinsics
        if os.path.exists(ExtPath):
            self.rvec, self.tvec = self.LoadExt(ExtPath)
        else:
            print("No extrinsics found, loading default")
            self.rvec = [0,0,0]
            self.tvec = [0,0,0]
            
        ##Load camera Intrinsics
        if os.path.exists(IntPath):
            self.camMat, self.distCoef = self.LoadInt(IntPath)
        else:
            print("No intrinsics found, loading default")
            self.camMat = np.identity(3)
            self.distCoef = [0,0,0,0,0]
        
    
    def LoadSyncArr(self, SyncArrPath):
        """Load Sync Array"""
        SyncArray = p.load(open(SyncArrPath,"rb"))
        return (SyncArray)
    
    def LoadExt(self, ExtPath):
        """Load Extrinsic parameters"""
        rvec, tvec = p.load(open(ExtPath,"rb"))
        return (rvec,tvec)
    
    def LoadInt(self, IntPath):
        """Load Intrinsic parameters"""
        cameraMatrix, distCoeffs= p.load(open(IntPath,"rb"))
        return (cameraMatrix,distCoeffs)
    
    def ReadCSV(self, Path):
        """Load 2D keypoint ground truth data"""
        df = pd.read_csv(Path)
        return df

    def load2DKeypoint(self, K2DPath):
        """Load 2D keypoint ground truth data"""
        self.Keypoint2D = pd.read_csv(K2DPath)
        
    def load2DFilterKeypoint(self, K2DPath):
        """Load 2D keypoint ground truth data after GESD filtering"""
        self.Keypoint2DFilter = pd.read_csv(K2DPath)
    
    def loadBBoxData(self, BBoxPath):
        """Load BBox ground truth data"""
        self.BBox = pd.read_csv(BBoxPath)
        
    def load3DKeypoint(self, K3DPath):
        """Load BBox ground truth data"""
        self.Keypoint3D = pd.read_csv(K3DPath)
        
        
    def Read2DKeypointData(self, df, frame, bird,Keypoints=None,StripName=False):
        """ 
        Given 2D Keypoint pandas dataframe, extract 2D keypoint data for all keypoints for a given bird ID
        Returns dictionary with keypoint name and point
        
        df: 2D Keypoints dataframe
        frame: frame number
        bird: BirdID
        Keypoints: List of desired keypoints, if None, just plot all
        StripName: Whether to strip Bird ID from key names
        
        returns:
        Dict: {Keypoint: [x,y]}
        
        """
        ObjectColumns = [name for name in df.columns if name.startswith(bird)]
        AllObjPoints = []
        AllObjNames = []
        if len(ObjectColumns) ==0:
            return None,None
        #Assuming all object column sequence goes from x then y,
        #append list of coordinates 2 at a time
        counter = 0
        ObjectPoints = [None]*2 #Vector for 1 single 2D point

        #Vicon Output named "Frame", custom features output named "frame"
        #test for it:
        if 'frame' in df.columns:
            FrameName = 'frame'
        elif 'Frame' in df.columns:
            FrameName = 'Frame'

        for col in ObjectColumns:
            Index = df.index[df[FrameName]==frame].to_list()[0]
            ObjectPoints[counter] = float(df[col].values[Index]) 
            counter += 1
            if counter == 2:#if 2 values are filled
                AllObjPoints.append(ObjectPoints)
                counter = 0 #reset counter
                ObjectPoints = [None]*2
                AllObjNames.append(col.strip("_y"))
        
        OutDict = {}   
        for i in range(len(AllObjNames)):
            if Keypoints is None: #default, read all
                OutDict[AllObjNames[i]] = AllObjPoints[i]
            else:
                for Key in Keypoints:
                    if AllObjNames[i].endswith(Key):
                        if StripName:
                            OutDict[Key] = AllObjPoints[i]
                        else:
                            OutDict[AllObjNames[i]] = AllObjPoints[i]
                
        # OutDict = {AllObjNames[i]:AllObjPoints[i] for i in range(len(AllObjNames)) if AllObjNames[i].endswith(Key) for Key in Keypoints}
        return(OutDict)
        
    def Read3DKeypointData(self, df, frame, bird,Keypoints = None,StripName = False):
        """ 
        Given 3D Keypoint pandas dataframe, extract 3D keypoint data for all keypoints for a given bird ID
        Returns dictionary with keypoint name and point
        
        [input]
        df: 3D Keypoints dataframe
        frame: frame number
        bird: BirdID
        Keypoints: List of desired keypoints
        StripName: Whether to strip Bird ID from key names
        
        [output]
        returns:
        Dict: {Keypoint: point}
        
        """
        ObjectColumns = [name for name in df.columns if name.startswith(bird)]
        AllObjPoints = []
        AllObjNames = []
        if len(ObjectColumns) ==0:
            return None,None
        #Assuming all object column sequence goes from x then y then z,
        #append list of coordinates 3 at a time
        counter = 0
        ObjectPoints = [None]*3 #Vector for 1 single 3D point

        #Vicon Output named "Frame", custom features output named "frame"
        #test for it:
        if 'frame' in df.columns:
            FrameName = 'frame'
        elif 'Frame' in df.columns:
            FrameName = 'Frame'

        for col in ObjectColumns:
            Index = df.index[df[FrameName]==frame].to_list()[0]
            ObjectPoints[counter] = float(df[col].values[Index]) 
            counter += 1
            if counter == 3:#if 3 values are filled
                AllObjPoints.append(ObjectPoints)
                counter = 0 #reset counter
                ObjectPoints = [None]*3
                AllObjNames.append(col.strip("_z"))
            
        OutDict = {}   
        for i in range(len(AllObjNames)):
            if Keypoints is None: #default, read all
                OutDict[AllObjNames[i]] = AllObjPoints[i]
            else:
                for Key in Keypoints:
                    if AllObjNames[i].endswith(Key):
                        if StripName:
                            OutDict[Key] = AllObjPoints[i]
                        else:
                            OutDict[AllObjNames[i]] = AllObjPoints[i]
        # OutDict = {AllObjNames[i]:AllObjPoints[i] for i in range(len(AllObjNames))}
        return(OutDict)
    
    def GetBBoxData(self, df, frame, bird):
        """Given bbox dataframe, frame number and birdID, get topleft (Start) and bottom right (End) corner of BBox
        
        [input]
        df: bbox Keypoints dataframe
        frame: frame number
        bird: BirdID
        
        [output]
        Start: Top left corner of bbox, [x,y]
        End: Bottom right corner of bbox, [x,y]
        
        """
        ObjectColumns = [name for name in df.columns if name.startswith(bird)]
        if len(ObjectColumns) ==0:
            return None,None

        Index = df.index[df["frame"]==frame].to_list()[0]
        
        Start = (df.loc[Index][ObjectColumns[0]],df.loc[Index][ObjectColumns[1]])
        End = (Start[0]+df.loc[Index][ObjectColumns[2]],Start[1]+df.loc[Index][ObjectColumns[3]])

        return Start,End
    
    def GetBBoxMidPoint(self, df, frame, bird):
        """Get midpoint of Bounding box for given frame/ bird"""
        Start, End = self.GetBBoxData(df, frame, bird)
        MidPoint = [round(Start[0])+((round(End[0])-round(Start[0]))/2),round(Start[1])+((round(End[1])-round(Start[1]))/2)]
        return MidPoint
        
def main():
    CamName = "Cam1"
    VideoPath = "/home/alexchan/Documents/SampleDatasets/01072022Pigeon10/UndistortedVideos/Cam1_C0003.MP4"
    SyncArrPath = "/home/alexchan/Documents/SampleDatasets/01072022Pigeon10/Data/Cam1_C0003_SyncArr.p"
    ExtPath = "/home/alexchan/Documents/SampleDatasets/01072022Pigeon10/Data/training_01072022_02_Cam1_extrinsic.p"
    IntPath = "/home/alexchan/Documents/SampleDatasets/01072022Pigeon10/Data/Cam1_intrinsics.p"
    
    TestCam = CameraObject(CamName,VideoPath,SyncArrPath,ExtPath,IntPath, (3840,2160))
    
    
if __name__ == "__main__":
    main()