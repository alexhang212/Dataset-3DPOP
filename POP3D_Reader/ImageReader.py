# !/usr/bin/env python3
"""JSON reader class to read images from images sampled from 3DPOP"""

import json
import cv2
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def getColor(keyPoint):
    if keyPoint.endswith("beak"):
        return (255, 0 , 0 )
    elif keyPoint.endswith("nose"):
        return (63,133,205)
    elif keyPoint.endswith("leftEye"):
        return (0,255,0)
    elif keyPoint.endswith("rightEye"):
        return (0,0,255)
    elif keyPoint.endswith("leftShoulder"):
        return (255,255,0)
    elif keyPoint.endswith("rightShoulder"):
        return (255, 0 , 255)
    elif keyPoint.endswith("topKeel"):
        return (128,0,128)
    elif keyPoint.endswith("bottomKeel"):
        return (203,192,255)
    elif keyPoint.endswith("tail"):
        return (0, 255, 255)
    else:
        # return (0,165,255)
        return (0,255,0)

class ImageReader:
    
    def __init__(self, JSONPath,DatasetPath, Type = "3D"):
        """
        Initialize JSON reader object
        JSONPath: path to json file
        DatasetPath: Path to dataset root directory to read images
        Type: 2D or 3D, based om which type was read     
        """
        
        with open(JSONPath) as f:
            self.data = json.load(f)
        
        self.DatasetPath = DatasetPath
        self.Type = Type
        self.Info = self.data["info"]
        self.PrintInfo()
        self.Annotations = self.data["Annotations"]
        
        
    def PrintInfo(self):
        print("Loading JSON...")
        print(self.Info["Description"])
        print("Collated by: %s on: %s" %(self.Info["Collated by"],self.Info["Date"]))
        print("Total Images: %s"%(self.Info["TotalImages"]))
        
    def Extract3D(self,index):
        """Extract 3D data"""
        return self.Annotations[index]["Keypoint3D"]
        
    def Extract2D(self,index):
        """Extract 2D data"""
        if self.Type == "3D":
            CameraInfoDict = self.Annotations[index]["CameraData"]
            Out = [[val for key,val in SubDict.items() if key == "Keypoint2D"][0] for SubDict in CameraInfoDict]
        else:
            Out = self.Annotations[index]["Keypoint2D"]
            
        return Out

    def ExtractBBox(self,index):
        if self.Type == "3D":
            CameraInfoDict = self.Annotations[index]["CameraData"]
            Out = [[val for key,val in SubDict.items() if key == "BBox"][0] for SubDict in CameraInfoDict]
        else:
            Out = self.Annotations[index]["BBox"]

        return Out
    
    def GetImagePath(self,index):
        if self.Type == "3D":
            CameraInfoDict = self.Annotations[index]["CameraData"]
            Out = [[val for key,val in SubDict.items() if key == "Path"][0] for SubDict in CameraInfoDict]
        else:
            Out = self.Annotations[index]["Path"]
        return Out
    
    def GetSequenceCode(self,FileName):
        return FileName.split("-")[0]
    
    def GetIntrinsics(self, index):
        
        #get sequence from file path        
        CamInfo = self.data["Annotations"][index]["CameraData"]
        
        SequenceCode = self.GetSequenceCode(os.path.basename(CamInfo[0]["Path"]))
        
        camMatList = []
        distCoefList= []
        
        for x in range(4): ##For 3D pop, always 4 cam
            Cam = x+1
            Intpath = os.path.join(self.DatasetPath,"Calibration","%s-Cam%s-Intrinsics.p"%(SequenceCode,Cam))
            camMat, distCoef = pickle.load(open(Intpath,"rb"))
            camMatList.append(camMat)
            distCoefList.append(distCoef)
            
        return camMatList, distCoefList

    def GetExtrinsics(self, index):
        #get sequence from file path        
        CamInfo = self.data["Annotations"][index]["CameraData"]
        
        SequenceCode = self.GetSequenceCode(os.path.basename(CamInfo[0]["Path"]))
        
        rvecList = []
        tvecList = []
        
        for x in range(4): ##For 3D pop, always 4 cam
            Cam = x+1
            Extpath = os.path.join(self.DatasetPath,"Calibration","%s-Cam%s-Extrinsics.p"%(SequenceCode,Cam))
            rvec, tvec = pickle.load(open(Extpath,"rb"))
            rvecList.append(rvec)
            tvecList.append(tvec)
            
        return rvecList, tvecList
    
    def GetGTArray(self,Indexes):
        """
        Get array of all annotation ground truth
        shape: (N,9,3)
        Indexes is list of index of image
        """
        GTList = []
        for i in Indexes:
           GT = np.array(list(self.Extract3D(i).values()))
           GTList.append(GT)
        
        GTArray = np.stack(GTList)
        # import ipdb;ipdb.set_trace()

        return GTArray
    
    def CheckAnnotations(self, index, show=True, jupyter = False):
        if self.Type == "3D":
            ImgPath = self.GetImagePath(index)[0]
            Key2D = self.Extract2D(index)[0]
            BBox = self.ExtractBBox(index)[0]
        else:
            ImgPath = self.GetImagePath(index)
            Key2D = self.Extract2D(index)
            BBox = self.ExtractBBox(index)
        # import ipdb;ipdb.set_trace()
        
        RealImgPath = os.path.join(self.DatasetPath,ImgPath)

        img = cv2.imread(RealImgPath)
        
        ##Draw keypoints:
        for BirdID,Key2DDict in Key2D.items():
            for key, pts in Key2DDict.items():
                point = (round(pts[0]),round(pts[1]))
                colour = getColor(key)
                cv2.circle(img,point,3,colour, -1)
            BBoxData = BBox[BirdID]
            cv2.rectangle(img,(round(BBoxData[0]),round(BBoxData[1])),(round(BBoxData[2]),round(BBoxData[3])),(255,0,0),3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, BirdID,(round(BBoxData[0]),round(BBoxData[1])), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        if show:
            cv2.imshow('image',img)
            cv2.waitKey(0)

        if jupyter:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.show()
            
        cv2.destroyAllWindows()
        return img


        
if __name__ == "__main__":
    JSONPath = "/home/alexchan/Documents/3D-MuPPET/TrainingData/N6000/Annotation/Test-3D.json"
    DatasetPath = "/home/alexchan/Documents/3D-MuPPET/TrainingData/N6000"
    Dataset = ImageReader(JSONPath,DatasetPath, Type = "3D")
    len(Dataset.Annotations)

    for i in range(len(Dataset.Annotations)):
        Dataset.CheckAnnotations(i,show=True, jupyter=False)

        