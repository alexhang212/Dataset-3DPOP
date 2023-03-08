# First prototype for the file to create annotation from the given image file and save them in respective feature

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





class imageAnnotationTool_Multi:

    def __init__(self, projectSettings):
        """
        Initialize the class for annotation of images
        :param videoFiles: Lise of video files
        :param path: path for reading features
        """
        settingsDict = projectSettings.settingsDict
        self.settingsDict = settingsDict

        videoFiles = settingsDict["videoFiles"]
        rootDir = settingsDict["rootDirectory"]
        # Iterate through path anf provide global path
        for file in range(len(videoFiles)):
            videoFiles[file] = os.path.join(rootDir, videoFiles[file])

        # import ipdb; ipdb.set_trace()
        
        print("Init class for image annotation")
        self.rootDir = rootDir
        self.videoFiles = videoFiles
        self.stepSize = int(settingsDict["stepSize"])
        self.startFrame = int(settingsDict["startFrame"])
        self.endFrame = int(settingsDict["endFrame"])
        self.frameInfoFile = os.path.join(settingsDict["DataDirectory"], settingsDict["framesToCaptureFile"])
        self.image = []
        self.features = rwOp.readFeaturesFromFile(settingsDict["customFeatureFile"])
        print("Current Custom Features:")
        print(self.features)
        
        self.subjects = settingsDict["Subjects"]
        self.dataPoints, self.dataBaseKeyPoints = self.createDatabase()
        self.point = [0,0]
        self.toggleAnnotationView = True
        self.Allfeatures = self.createAdvancedFeatures()

        # import ipdb; ipdb.set_trace()
        #read in vicon info, depending on custom mode or vicon mode
        SystemData = system.SystemInit(projectSettings,rootDir)

        
        # viconSystemData.printSysteInfo()
        self.CoordData = SystemData.CoordData #coordinate csv from trial
        self.viconCamObjects = SystemData.viconCameraObjects
        self.imageObjects = SystemData.viconImageObjects
        self.dataDir = SystemData.dataDir

        
        # test
        self.frameToAnnotate = self.loadFrameInfo()
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
        for subject in self.subjects:
            for feature in self.features:
                defaultDataSeries[str(subject) + "_"+ str(feature)+"_x"] = 0
                defaultDataSeries[str(subject) + "_"+ str(feature)+"_y"] = 0

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
            for featureName in self.Allfeatures:
                keyPoint = self.getkeypointInfo(featureName)
                color = self.getColor(featureName)
                if keyPoint != [0, 0] :
                    # cv.circle(image, keyPoint, 1, color, -1)

                    #Other way of plotting/;
                    cv.circle(image, keyPoint, 5, color, 1)
                    cv.line(image, (keyPoint[0] - 5, keyPoint[1]),(keyPoint[0]+5,keyPoint[1]),color , 1)
                    cv.line(image, (keyPoint[0], keyPoint[1] - 5), (keyPoint[0] , keyPoint[1]+ 5), color, 1)
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
        cv.putText(image, "Frame No :" + str(frameNo), (500, 100), font, 1,
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
            for feature in self.Allfeatures:
                self.setKeypointInfo(feature, self.point)
        else: ##Added subject param, only removes points of specific subject when specified
            self.point = [0, 0]
            for feature in self.Allfeatures:
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

    def run(self):
        
        if self.settingsDict["Mode"]=="Custom":
            custom = True
        else:
            custom = False


        # Creating video objects
        videoObjects = []

        for i in range(len(self.videoFiles)):
            # import ipdb; ipdb.set_trace()
            CameraObj = self.viconCamObjects[i]
            ImgObj = self.imageObjects[i]
            # destroy pre existing windows
            cv.destroyAllWindows()
            videoObjects.append(vid.VideoVicon(self.videoFiles[i], i))
            windowName = videoObjects[i].__getattribute__("windowName")
            # import ipdb;ipdb.set_trace()
            csvFileName = os.path.join(self.dataDir, self.settingsDict["annotationFiles"][i])

            if os.path.exists(csvFileName):
                self.loadDatabaseFromcsv(csvFileName)
                print("File already exists, loading database from file")
            else:
            # reset the database entries
               self.dataBaseKeyPoints = self.dataBaseKeyPoints.iloc[0:0]
            # Reset annotations before starting annotations, it is not carried from one video to another.
               self.resetAnnotations()

            mode = "unselected"
            cv.namedWindow(windowName, cv.WINDOW_NORMAL)
            cv.setMouseCallback(windowName, unselected , mode)
            # For bounding box view
            cv.namedWindow("bBoxWindow", cv.WINDOW_NORMAL)

            # Finding no of frames to run the algorithm
            frameCount = videoObjects[i].totalFrameCount
            frameNo = self.startFrame

            if self.startFrame == self.endFrame or self.startFrame > self.endFrame or self.endFrame == 0:
                print("Start and end frames given same therefore ")
                self.endFrame = frameCount

            #commented this out, sony messing things up because of diff video frame lengths
            # assert (self.endFrame <= frameCount), "Video sequence has less frames than defined as end frame in settings"

            # Prepare the list of files to be annotated either from the annotation selector or create custom list based on
            # the start, end and steps information given in the settings file.
            if len(self.frameToAnnotate):
                print("Preparing frames for annotation based on the selected files through annotation selector tool.")
                listOfFrames = self.frameToAnnotate
                print(listOfFrames)
            else: # else check out frames with given start and end sequence
                print("Preparing frames for annotation using the settings file.")
                listOfFrames = list(range(self.startFrame, self.endFrame, self.stepSize))
                print(listOfFrames)
                
            # import ipdb; ipdb.set_trace()

            ##loop per subject
            for subject in self.subjects: 
                # Load information about annotation if it exists in the .csv file
                id = 0
                self.loadFromDatabase(int(listOfFrames[id]))

                while 0 <= id < len(listOfFrames):
                    # import ipdb; ipdb.set_trace()
                    # print(subject)
                    # print(id)
                    if custom:
                        frameNo = int(listOfFrames[id])-CameraObj.FrameDiff
                    else:
                        frameNo = int(listOfFrames[id])
                    image = videoObjects[i].getFrame(frameNo)
                    clone = image.copy()
                    # import ipdb; ipdb.set_trace()
                    ##Get subject approx position and draw bounding box around it to identify bird
                    
                    #for custom sony cameras
                    if self.settingsDict["Mode"] == "Custom":
                        ViconFrame = FindClosestFrame(CameraObj.SyncArray, frameNo)
                        # import ipdb; ipdb.set_trace()
                        coordinateDict = self.CoordData.getCoordDataForViconFrame(int(ViconFrame))
                    else:
                        coordinateDict = self.CoordData.getCoordDataForVideoFrame(int(listOfFrames[id]))
                    object = subject + "_bp" #Using backpack as arbitiary point to generate bbox
                    ObjectCoordDict = fileOp.get3DDictofObject(coordinateDict, object)

                    

                    featureDictCamSpace = CameraObj.transferFeaturesToObjectSpace(ObjectCoordDict,custom=True)

                    
                    CameraObj.setFeatures(featureDictCamSpace)
                    imageFeaturesDict = ImgObj.projectFeaturesFromCamSpaceToImageSpace(featureDictCamSpace)

                    #choosing Arbitiary point: point 1 of backpack
                    ArbPoint = list(imageFeaturesDict.values())[0]
                    # print(ArbPoint)
                    height, width, channel = clone.shape

                    #Draw arbitiary region of interest around target subject:
                    # if (self.point[0] != 0 and self.point[1] != 0):
                    if math.isnan(ArbPoint[0]) or ArbPoint[0]>width or ArbPoint[1]>height:
                        #if point is NAN, or outside frame, not tracked, wont use this frame
                        self.updateDataBase(int(listOfFrames[id]),csvFileName)
                        id += 1
                        mode = "unselected"
                        if id < len(listOfFrames):
                            # frameNo = int(listOfFrames[id])
                            self.loadFromDatabase(int(listOfFrames[id]))
                        continue

                    roi = getRoiPoint(ArbPoint, height, width, 120)
                    roiWidth = roi[3] - roi[2] # x2 - x1
                    roiHeight = roi[1] - roi[0] # y2 - y1
                    # import ipdb; ipdb.set_trace()

                    #Draw bounding box
                    DrawBBox(clone, roi, roiWidth, roiHeight)
                    ObjectFeatName = "unselected"
                    #define subject specific point based on mode
                    if mode is not "unselected":
                        ObjectFeatName = "%s_%s"%(subject,mode)


                    #Loops around:
                    # Update points
                    self.setKeypointInfo(ObjectFeatName, self.point)
                    # Draw points on the image
                    self.drawKeypoints(clone, ObjectFeatName, frameNo)

                    #get roi window
                    # import ipdb; ipdb.set_trace()
                    roiImage = clone[roi[2]:roi[3] , roi[0]:roi[1], :]
                    resized = cv.resize(roiImage, (400,400), interpolation = cv.INTER_AREA)
                    # clone[0:resized.shape[0], 0:resized.shape[1]] = resized
                    #show image
                    cv.imshow("bBoxWindow", np.array(resized))
                    cv.imshow(windowName, clone)

                    k = cv.waitKey(10)

                    keys = [ord(str(x)) for x in list(self.annotationDict)]

                    if k in keys: #if keypress matches unicode key
                        index = keys.index(k)
                        print("Mode", self.annotationDict[index])
                        mode = self.annotationDict[index]
                        if mode != "unselected":
                            ObjectFeatName = "%s_%s"%(subject,mode)
                            self.point = [self.getkeypointInfo(ObjectFeatName)[0], self.getkeypointInfo(ObjectFeatName)[1]]
                            cv.setMouseCallback(windowName, self.pointAssignment, self.point)
                        else:
                            cv.setMouseCallback(windowName, unselected, self.point)

                    if k == ord('w'):
                        self.point[1] -= 1 # Subtract y coordinate

                    if k == ord('a'):
                        self.point[0] -= 1  # Subtract x coordinate

                    if k == ord('s'):
                        self.point[1] += 1  # add 1 unit y coordinate

                    if k == ord('d'):
                        self.point[0] += 1  # add 1 unit x coordinate

                    if k == ord('r'): # Erase annotations
                        self.resetAnnotations(subject)

                    if k == ord('h'): # Help print key info
                        self.printAnnotationInfo()

                    if k == ord('i'): # Info
                        self.toggleAnnotationView = not self.toggleAnnotationView

                    if k == ord('q'):  # Quit
                        #Save final data
                        self.updateDataBase(int(listOfFrames[id]), csvFileName)
                        # Clear the data
                        # self.dataBaseKeyPoints = self.dataBaseKeyPoints.iloc[0:0]
                        # destroy the existing video window
                        cv.destroyAllWindows()
                        break

                    if k == ord('b'): # Go back
                        # make mode unselected
                        mode = "unselected"
                        if id != 0:
                            id -= 1
                            # frameNo = int(listOfFrames[id])
                            self.loadFromDatabase(int(listOfFrames[id]))
                        else:
                            print("Can not go back. Frame : ", frameNo, "\n Start Frame : ", self.startFrame )

                    if k == ord('n'): # Next image
                        self.updateDataBase(int(listOfFrames[id]),csvFileName)
                        id += 1
                        mode = "unselected"
                        if id < len(listOfFrames):
                            # frameNo = int(listOfFrames[id])
                            self.loadFromDatabase(int(listOfFrames[id]))
                        continue
                
            cv.destroyAllWindows() #once out of subject loop, destroy all windows
            #done with 1 camera, go next


if __name__ == "__main__":

    # Example will use this class to make small application of viewing the video files
    settingsObject = settingsGenerator.xmlSettingsParser("D:\\BirdTrackingProject\\20190618_PigeonPostureDataset\\settings_session02.xml")
    annotator = imageAnnotationTool_Multi(settingsObject.settingsDict)
    annotator.run()
