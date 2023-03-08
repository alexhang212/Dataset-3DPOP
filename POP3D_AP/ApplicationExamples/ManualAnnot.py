# !/usr/bin/env python3
"""
Prepares dataset for manual annoatation for reviewer comments

Also have new image annotation tool for images

"""

import sys
sys.path.append('./')
sys.path.append('../')

import random
random.seed(10)

import cv2 as cv
import numpy as np
from System import videoVicon as vid
from FileOperations import rwOperations as rwOp
from FileOperations import settingsGenerator
from System import systemInit as system
import pandas as pd
import os
from FileOperations import rwOperations as fileOp
import math
from POP3D_Reader import Trial, CameraObject
from tqdm import tqdm
import re

def PlotLine(MarkerDict,Key1Short,Key2Short,Colour,img):
    """
    Plot a line in opencv between Key1 and Key2
    Input is a dictionary of points
    
    """
    
    #Get index for given keypoint
    Key1Name = [k for k in list(MarkerDict.keys()) if Key1Short in k][0]
    Key2Name = [k for k in list(MarkerDict.keys())  if Key2Short in k][0]
    
    pt1 = MarkerDict[Key1Name]
    pt2 = MarkerDict[Key2Name]

    if np.isnan(pt1[0]) or math.isinf(pt1[0]) or math.isinf(pt1[1]):
        return None
    elif np.isnan(pt2[0]) or math.isinf(pt2[0]) or math.isinf(pt2[1]):
        return None

    point1 = (round(pt1[0]),round(pt1[1]))
    point2 = (round(pt2[0]),round(pt2[1]))

    cv.line(img,point1,point2,Colour,2 )


def setPoint(point,x,y):
    point[0] = x
    point[1] = y

def unselected(event, x, y, flags, param ):
    if event == cv.EVENT_LBUTTONDOWN:
        print("No mode selected")

def boundingBox(event, x, y, flags, point ):
    if event == cv.EVENT_LBUTTONDOWN:
        point["x"] = x
        point["y"] = y
    if event == cv.EVENT_RBUTTONDOWN:
        point["width"] = x - point["x"]
        point["height"] = y - point["y"]


def getRoiPoint(point, height, width , size = 200):
    """
    Author Alex Chan
    Get ROI from a point
    """
    x = int(point[0])
    y= int(point[1])

    roi = [0 , 0, 200, 200]

    roi[0] = x - size
    if roi[0] < 0:
        roi[0] = 0

    roi[1] = x + size
    if roi[1] > width:
        roi[1] = width - 1

    roi[2] = y - size
    if roi[2]< 0:
        roi[2] = 0

    roi[3]= y + size
    if roi[3] > height:
        roi[3] = height -1

    return roi

def DrawBBox(img, roi, roiWidth, roiHeight):
    """
    Function to draw bounding box
    """

    cv.rectangle(img,(roi[1],roi[2]),(roi[0],roi[3]),(255,0,0),2)

def FindClosestFrame(SyncArray, Frame):
    """
    Author: Alex Chan

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






class imageAnnotationTool_Manual:

    def __init__(self,ImageDir,GroundTruthCSV, OutputCSV,FeatureFile ):
        """
        Initialize the class for annotation of images

        """
        self.ImageDir = ImageDir
        self.GroundTruthCSV = pd.read_csv(GroundTruthCSV)
        self.OutputCSV = OutputCSV
        self.features = self.readFeaturePointsFromFile(FeatureFile)
        
        self.dataPoints, self.dataBaseKeyPoints = self.createDatabase()
        self.point = [0,0]
        self.toggleAnnotationView = True
        self.createAnnotationDict()

    def createAnnotationDict(self):
        self.annotationDict = {0:"unselected"}
        for id in range(1,len(self.features)+1):
            self.annotationDict[id] = self.features[id-1]

    def pointAssignment(self, event, x, y, flags, point):
        if event == cv.EVENT_LBUTTONDOWN:
            setPoint(point, x, y)
        elif event == cv.EVENT_RBUTTONDOWN:
            setPoint(point, 0, 0)

    def loadFrameInfo(self):
        """
        Loads the information about frames to use for a file
        """
        # assert (os.path.exists(self.frameInfoFile) == True), "File does not exist"
        framesToAnnotate = []
        if not os.path.exists(self.frameInfoFile):
            print("File does not exist.")
            return framesToAnnotate

        # Open file and read annotation frames
        file = open(self.frameInfoFile)
        lines = [line.rstrip('\n') for line in file]
        # Read point information from the file
        for line in lines:
            if line.isdigit():
                framesToAnnotate.append(line)

        return framesToAnnotate

    def createAdvancedFeatures(self):
        """
        Author: Alex Chan
        Create longer list of all features for each individual pigeon
        
        """
        featureList = []
        
        for subject in self.subjects:
            for feature in self.features:
                featureList.append(str(subject) + "_"+ str(feature))

        return featureList



    def createDatabase(self):
        """
        Create data base related structure
        :return:
        """
        defaultDataSeries = {"frame":0}
        for feature in self.features:
            defaultDataSeries[str(feature)+"_x"] = 0
            defaultDataSeries[str(feature)+"_y"] = 0

        dataFrame = pd.DataFrame(columns = list(defaultDataSeries))


        return defaultDataSeries,dataFrame

    def readFeaturePointsFromFile(self, path):
        """
        Loads features from the given file path
        :param : Path to text file with all required featues
        :return: return list of features
        """
        features = []
        assert (os.path.exists(path) == True), "File for custom features does not exist"
        if os.path.exists(path):
            file = open(path)
            lines = [line.rstrip('\n') for line in file]
            # Read point information from the file
            for line in lines:
                features.append(str(line))

        return features


    def getkeypointInfo(self, str):
        if (str is not "unselected"):
            x = self.dataPoints[str+"_x"]
            y = self.dataPoints[str+"_y"]
            return x,y

    def setKeypointInfo(self, str, point):
        if (str is not "unselected"):
            self.dataPoints[str + "_x"] = point[0]
            self.dataPoints[str + "_y"] = point[1]
            return True
        else:
            return False

    def getColor(self, keyPoint):
        # import ipdb; ipdb.set_trace()
        if keyPoint.endswith(self.features[0]):
            return (255, 0 , 0 )
        elif keyPoint.endswith(self.features[1]):
            return (63,133,205)
        elif keyPoint.endswith(self.features[2]):
            return (0,255,0)
        elif keyPoint.endswith(self.features[3]):
            return (0,0,255)
        elif keyPoint.endswith(self.features[4]):
            return (255,255,0)
        elif keyPoint.endswith(self.features[5]):
            return (255, 0 , 255)
        elif keyPoint.endswith(self.features[6]):
            return (128,0,128)
        elif keyPoint.endswith(self.features[7]):
            return (203,192,255)
        elif keyPoint.endswith(self.features[8]):
            return (0, 255, 255)
        else:
            return (0,165,255)

    def drawKeypoints(self, image, mode, frameNo):
        # Draw all the key points if the annotation view is True
        if self.toggleAnnotationView:
            for featureName in self.features:
                keyPoint = self.getkeypointInfo(featureName)
                color = self.getColor(featureName)
                if keyPoint != [0, 0] :
                    cv.circle(image, keyPoint, 1, color, -1)

                    #Other way of plotting/;
                    # cv.circle(image, keyPoint, 5, color, 1)
                    # cv.line(image, (keyPoint[0] - 5, keyPoint[1]),(keyPoint[0]+5,keyPoint[1]),color , 1)
                    # cv.line(image, (keyPoint[0], keyPoint[1] - 5), (keyPoint[0] , keyPoint[1]+ 5), color, 1)
        else: # Draw only require feature point if only current view is required
            keyPoint = self.getkeypointInfo(mode)
            color = (255,255,255)
            if keyPoint != [0,0] and mode != "unselected":
                cv.circle(image, keyPoint, 1, color, -1)
                # cv.line(image, (keyPoint[0] - 2, keyPoint[1]), (keyPoint[0] + 2, keyPoint[1]), color, 1)
                # cv.line(image, (keyPoint[0], keyPoint[1] - 2), (keyPoint[0], keyPoint[1] + 2), color, 1)

        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(image, "Annotation Mode :" + mode , (500, 50), font, 1,
                    (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(image, "Image No :" + str(frameNo), (500, 100), font, 1,
                   (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(image, "Press 'h' for help", (500, 250), font, 1,
                (255, 255, 255), 2, cv.LINE_AA)


    def getRoi(self, height, width , size = 200):
        x = self.point[0]
        y = self.point[1]
        roi = [0 , 0, 200, 200]

        roi[0] = x - size
        if roi[0] < 0:
            roi[0] = 0

        roi[1] = x + size
        if roi[1] > width:
            roi[1] = width - 1

        roi[2] = y - size
        if roi[2]< 0:
            roi[2] = 0

        roi[3]= y + size
        if roi[3] > height:
            roi[3] = height -1

        return roi

    def printAnnotationInfo(self):
        print("Keybord Commands")
        print("w-s-a-d : Move one pixel in relative direction")
        print("h : help information")
        print("q : Quit")
        print("b : back")
        print("r : reset annotations")
        print("n : Next frame")
        for id in self.annotationDict:
            print(id , ":", self.annotationDict[id])

    def saveData(self, filePath): # Saving databaseto csv file
        self.dataBaseKeyPoints.to_csv(filePath, index = False)

    def updateDataBase(self, frameNo, csvFileName):
        self.dataPoints["frame"] = frameNo
        # if entry exists we just modify the existing entry
        list = self.dataBaseKeyPoints["frame"].tolist()
        if frameNo in self.dataBaseKeyPoints["frame"].tolist():
            index = self.dataBaseKeyPoints["frame"].tolist().index(frameNo)
            self.dataBaseKeyPoints.loc[index] = self.dataPoints
        else : # if entry does not exist we add a new
            self.dataBaseKeyPoints = self.dataBaseKeyPoints.append(self.dataPoints, ignore_index=True)

        self.saveData(csvFileName)

    def resetAnnotations(self, subject=None):
        if subject is None:
            self.point = [0, 0]
            for feature in self.features:
                self.setKeypointInfo(feature, self.point)
        else: ##Added subject param, only removes points of specific subject when specified
            self.point = [0, 0]
            for feature in self.features:
                if feature.startswith(subject):
                    self.setKeypointInfo(feature, self.point)
                else:
                    continue

        print("Erasing all points : ", self.dataPoints)

    def loadDatabaseFromcsv(self,filename):
        """
        Given file name load database from existing csv file
        :param filename: str (.csv)
        :return: None
        """
        self.dataBaseKeyPoints = pd.read_csv(filename, index_col= False)
        if self.dataBaseKeyPoints.empty:
            print("Database from loaded file is empty : ", filename)

    def loadFromDatabase(self, frameNo):
        self.dataPoints["frame"] = frameNo
        if frameNo in self.dataBaseKeyPoints["frame"].tolist():
            index = self.dataBaseKeyPoints["frame"].tolist().index(frameNo)
            dataSeries = self.dataBaseKeyPoints.loc[index]
            self.dataPoints = dataSeries.to_dict()
            print("Method 1", self.dataPoints)
        else:
            self.resetAnnotations()
        return

    def run(self, startFrame):
        """Run Annotation"""
        # import ipdb;ipdb.set_trace()
        counter = startFrame #image coutner
        GTdf = self.GroundTruthCSV
        OutputCSV = self.OutputCSV
        
        if os.path.exists(OutputCSV):
            self.loadDatabaseFromcsv(OutputCSV)
            print("File already exists, loading database from file")
        else:
        # reset the database entries
            self.dataBaseKeyPoints = self.dataBaseKeyPoints.iloc[0:0]
        # Reset annotations before starting annotations, it is not carried from one video to another.
            self.resetAnnotations()
            
        ##Annotate:
        mode = "unselected"
        cv.destroyAllWindows()

        cv.namedWindow("Window", cv.WINDOW_NORMAL)
        cv.namedWindow("bBoxWindow", cv.WINDOW_NORMAL)

        cv.setMouseCallback("Window", unselected , mode)
        # For bounding box view

        # import ipdb; ipdb.set_trace()
        self.loadFromDatabase(counter)

        while 0 <= counter < len(GTdf.index):
            # import ipdb; ipdb.set_trace()
            ImgRow = GTdf.iloc[counter]
            ImgPath = os.path.join(self.ImageDir,ImgRow["ImageName"])

            image = cv.imread(ImgPath)
            clone = image.copy()
            # import ipdb; ipdb.set_trace()

            #choosing Arbitiary point to draw bbox
            ArbPoint = GTdf[["bp1_x", "bp1_y"]].iloc[counter].to_list()
            # print(ArbPoint)
            height, width, channel = clone.shape
            
            roi = getRoiPoint(ArbPoint, height, width, 120)
            roiWidth = roi[3] - roi[2] # x2 - x1
            roiHeight = roi[1] - roi[0] # y2 - y1
            
            if roiWidth < 0:
                counter += 1
                continue

            # import ipdb; ipdb.set_trace()

            #Draw bounding box
            DrawBBox(clone, roi, roiWidth, roiHeight)
            # import ipdb;ipdb.set_trace()


            ObjectFeatName = "unselected"
            #define subject specific point based on mode
            if mode is not "unselected":
                ObjectFeatName = mode


            #Loops around:
            # Update points
            self.setKeypointInfo(ObjectFeatName, self.point)
            # Draw points on the image
            self.drawKeypoints(clone, ObjectFeatName, counter)
            
            
            roiImage = clone[roi[2]:roi[3] , roi[0]:roi[1], :]
            resized = cv.resize(roiImage, (400,400), interpolation = cv.INTER_AREA)
            
            # clone[0:resized.shape[0], 0:resized.shape[1]] = resized
            #show image
            cv.imshow("Window", clone)
            cv.imshow("bBoxWindow", np.array(resized))

            k = cv.waitKey(10)

            # import ipdb;ipdb.set_trace()

            keys = [ord(str(x)) for x in list(self.annotationDict)]

            if k in keys: #if keypress matches unicode key
                index = keys.index(k)
                print("Mode", self.annotationDict[index])
                mode = self.annotationDict[index]
                if mode != "unselected":
                    ObjectFeatName = mode
                    self.point = [self.getkeypointInfo(ObjectFeatName)[0], self.getkeypointInfo(ObjectFeatName)[1]]
                    cv.setMouseCallback("Window", self.pointAssignment, self.point)
                else:
                    cv.setMouseCallback("Window", unselected, self.point)

            if k == ord('w'):
                self.point[1] -= 1 # Subtract y coordinate

            if k == ord('a'):
                self.point[0] -= 1  # Subtract x coordinate

            if k == ord('s'):
                self.point[1] += 1  # add 1 unit y coordinate

            if k == ord('d'):
                self.point[0] += 1  # add 1 unit x coordinate

            if k == ord('r'): # Erase annotations
                self.resetAnnotations()

            if k == ord('h'): # Help print key info
                self.printAnnotationInfo()

            if k == ord('i'): # Info
                self.toggleAnnotationView = not self.toggleAnnotationView

            if k == ord('q'):  # Quit
                #Save final data
                self.updateDataBase(counter, OutputCSV)
                # Clear the data
                # destroy the existing video window
                cv.destroyAllWindows()
                break

            if k == ord('b'): # Go back
                # make mode unselected
                mode = "unselected"
                if counter != 0:
                    counter -= 1
                    # frameNo = int(listOfFrames[id])
                    self.loadFromDatabase(counter)
                else:
                    print("Can not go back. Frame : ", counter, "\n Start Frame : ", self.startFrame )

            if k == ord('n'): # Next image
                self.updateDataBase(counter,OutputCSV)
                counter += 1
                mode = "unselected"
                if counter < len(GTdf.index):
                    # frameNo = int(listOfFrames[id])
                    self.loadFromDatabase(counter)
                continue
        
    cv.destroyAllWindows() #once out of subject loop, destroy all windows
    #done with 1 camera, go next




def PrepareImages(BaseDir,ImageDir,dataDir):
    """Go through dataset, choose random frames from each trial, save as image, while saving ground truth in csv"""
    
    GTDictFull = {}
    counter = 0

    while counter < 1000: #Prepare 1000+ images    
        for i in range(59):
            Sequence = i+1
            PigeonTrial = Trial.Trial(dataDir , Sequence)
        
            PigeonTrial.load3DPopDataset(Filter = True)

            for Cam in PigeonTrial.camObjects:
                print(counter)
                Key2D = Cam.Keypoint2D
                Key2D = Key2D.dropna()
                
                #sample random row and random individual
                RandomRow = Key2D.sample(1)
                RowDict = RandomRow.to_dict('list')
                RandomID =  random.sample(PigeonTrial.Subjects, 1)
                
                GTDict = {"_".join(k.split("_")[2:]):v[0] for k,v in RowDict.items() if k.startswith(RandomID[0])}
                GTDict["frame"] = RowDict["frame"][0]
                GTDict["cam"] = Cam.CamName
                GTDict["Trial"] = PigeonTrial.TrialName
                GTDict["ImageName"] = "Image%s.jpg"%counter
                GTDictFull.update({str(counter):GTDict})
                
                #Save image:
                VideoPath = Cam.VideoPath
                cap = cv.VideoCapture(VideoPath)
                cap.set(cv.CAP_PROP_POS_FRAMES,int(GTDict["frame"])) 
                ret, frame = cap.read()
                if ret==True:
                    cv.imwrite(os.path.join(ImageDir,"Image%s.jpg"%counter), frame)
                ####
                
                counter +=1                
            
    GTDFFull = pd.DataFrame.from_dict(GTDictFull, orient="index")
    GTDFFull.to_csv(os.path.join(BaseDir, "GroundTruthData.csv"))
    return(GTDFFull)

def Calc_edErr_2D(pt1,pt2):
    """Calculate euclidian error between 2 points"""
    return math.sqrt(((pt2[0]-pt1[0])**2) +((pt2[1]-pt1[1])**2) )

def ProcessFrameDict(Dict):
    """Provess frame wise dict to 2d dict per keypoint"""
    OutDict = {}
    counter = 0
    Point2D =[None]*2

    for key,val in Dict.items():
        # import ipdb;ipdb.set_trace()
        if key.endswith("_x") or key.endswith("_y"):
            if val[0] == 0:
                Point2D[counter] = math.nan
            else:
                Point2D[counter] = val[0]
            counter +=1

            if counter ==2:
                Name = key.strip("_y")
                OutDict[Name] = Point2D
                Point2D = [None]*2
                counter = 0
                # KeypointNameList.append(key.strip("_z"))
        
    return OutDict



def CalcError(GroundTruthCSV,AnnotCSV, ErrorCSV):

    GTdf = pd.read_csv(GroundTruthCSV)
    Annotdf = pd.read_csv(AnnotCSV)

    ErrorDict = {}

    for i in range(len(Annotdf.index)):
        
        GTDict = GTdf.loc[GTdf["ImageName"]=="Image%i.jpg"%i].to_dict("list")
        DFDict = Annotdf.loc[Annotdf["frame"]==i].to_dict("list")
        
        if len(DFDict["frame"]) == 0: #no frame information
            continue
        
        GTDict2D = ProcessFrameDict(GTDict)
        DFDict2D = ProcessFrameDict(DFDict)
        OutDict = {}

        ##Calc average error:
        for key,val in DFDict2D.items():
            if np.isnan(val).any():
                OutDict.update({key:np.nan})
            else:
                Error = Calc_edErr_2D(GTDict2D[key],DFDict2D[key])
                OutDict.update({key:Error})
                
        OutDict["frame"] = i
        ErrorDict.update({str(i):OutDict})
    
    ErrorDF = pd.DataFrame.from_dict(ErrorDict, orient="index")

    ErrorDF.to_csv(ErrorCSV)
    ##GET RMSE Errors:
    RMSEErrorList = []
    for col in ErrorDF.columns:
        ColData = ErrorDF[col].to_list()
        ColDataClean = [x for x in ColData if np.isnan(x) == False]
        
        if len(ColDataClean) == 0 or col == "frame":
            continue
        
        RMSEError = np.sqrt(np.mean(np.array(ColDataClean)**2))
        RMSEErrorList.append(RMSEError)
        print(col)
        print("RMSE: %s"%RMSEError)
    
    AllRMSE = np.sqrt(np.mean(np.array(RMSEErrorList)**2))
    print("RMSE of all points: %f"%AllRMSE)


def ExtractPCK(GroundTruthCSV,ErrorCSV,dataDir,PCKCSV):
    """
    Extract PCK05 and PCK10: stands for % points within 5%/10% of boundingbox width
    """
    
    GTdf = pd.read_csv(GroundTruthCSV)
    Errordf = pd.read_csv(ErrorCSV)
    PCKDict = {}
    for i in tqdm(range(len(Errordf.index))):
        SequenceString = GTdf.iloc[i]["Trial"].split("_")[0]
        SequenceNum = int(re.findall(r'\d+',SequenceString)[0])
        print(SequenceNum)
        PigeonTrial = Trial.Trial(dataDir,SequenceNum)
        PigeonTrial.load3DPopDataset(Filter=True)
    
        CamIndex = int(GTdf.iloc[i]["cam"][-1])-1
        Frame = GTdf.iloc[i]["frame"]
        BBoxdf = PigeonTrial.camObjects[CamIndex].BBox
        
        ##Forgot to save individual name, extracting from 2D keypoint df
        Key2Ddf = PigeonTrial.camObjects[CamIndex].Keypoint2D
        FakePoint = GTdf.iloc[i]["bp1_x"]
        RowDict = Key2Ddf.loc[Key2Ddf["frame"] == Frame].to_dict("list")
        
        MatchingKeys = [k for k,v in RowDict.items() if v ==FakePoint ]
        if len(MatchingKeys) != 1:
            import ipdb;ipdb.set_trace()
            raise Exception("something wrong from searching id name")
        
        BirdID = "_".join(MatchingKeys[0].split("_")[0:2])
        
        TopLeft, BottomRight = PigeonTrial.camObjects[CamIndex].GetBBoxData(BBoxdf,Frame,BirdID)
        
        BBoxWidth = BottomRight[0] - TopLeft[0] 
        
        #Get PCKs:
        ErrordfRowDict = Errordf.iloc[i].to_dict()
        OutDict = {}
        for key, val in ErrordfRowDict.items():
            if np.isnan(val):
                continue
            PercentageError = (val/BBoxWidth)*100
            OutDict.update({"%s_PercentWidth"%key:PercentageError})
        PCKDict.update({str(i):OutDict})
        
    PCKdf = pd.DataFrame.from_dict(PCKDict, orient="index")
    PCKdf.to_csv(PCKCSV)
    
    #Calculate PCK:
    PCK05List = []
    PCK10List = []

    for col in PCKdf.columns:
        if col.startswith("bp"):
            ColData = PCKdf[col].to_list()
            ColDataClean = [x for x in ColData if np.isnan(x) == False]
            
            if len(ColDataClean) == 0 or col == "frame":
                continue
            # import ipdb;ipdb.set_trace()
            
            PCK05 = len([x for x in ColDataClean if x<5])/len(ColDataClean)
            PCK10 = len([x for x in ColDataClean if x<10])/len(ColDataClean)

            PCK05List.append(PCK05)
            PCK10List.append(PCK10)
            
            print(col)
            print("PCK05: %s"%PCK05)
            print("PCK10: %s"%PCK10)

        else:
            continue
        
    print("Mean PCK 05: %s"%(sum(PCK05List)/len(PCK05List)))
    print("Mean PCK 10: %s"%(sum(PCK10List)/len(PCK10List)))

def main():
    # BaseDir = "/media/alexchan/My Passport/Pop3DDataset/ManualAnnotations"
    ImageDir = "/media/alexchan/My Passport/Pop3DDataset/ManualAnnotations/Images/"
    dataDir = "/media/alexchan/My Passport/Pop3D-Dataset_Final/"
    
    # PrepareImages(BaseDir,ImageDir,dataDir)
    ##Annotation
    GroundTruthCSV = "/media/alexchan/My Passport/Pop3DDataset/ManualAnnotations/GroundTruthData.csv"
    OutputCSV =  "/media/alexchan/My Passport/Pop3DDataset/ManualAnnotations/AnnotationData.csv"
    FeatureFile =  "/media/alexchan/My Passport/Pop3DDataset/ManualAnnotations/customFeatures.txt"
    AnnotTool = imageAnnotationTool_Manual(ImageDir,GroundTruthCSV, OutputCSV,FeatureFile)
    # AnnotTool.run(977)
    
    #calculate error
    ErrorCSV = "/media/alexchan/My Passport/Pop3DDataset/ManualAnnotations/FinalError.csv"
    # CalcError(GroundTruthCSV,OutputCSV,ErrorCSV)
    
    PCKCSV = "/media/alexchan/My Passport/Pop3DDataset/ManualAnnotations/PCKError.csv"
    ExtractPCK(GroundTruthCSV,ErrorCSV,dataDir,PCKCSV)
    
if __name__ == "__main__":
    main()