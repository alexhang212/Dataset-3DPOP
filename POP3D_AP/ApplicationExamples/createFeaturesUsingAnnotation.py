"""
This file shows an example of using the custom 2D annotations and triangulating them to create 3D annotations.

"""


from logging import raiseExceptions
import cv2 as cv
import numpy as np
from Math import transformations as tf
import os
from System import systemInit as system
from FileOperations import rwOperations as fileOp
from System import objectVicon as viconObj
from System import videoVicon as videoObj
from System import imageVicon as imageObj
from Math import stereoComputation as stereo
import pandas as pd
from FileOperations import settingsGenerator
from tqdm import tqdm


#ignoring pandas future warning for append
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

# Debug Function to print the calibration Information
def printCalibInformation(camInstances):
    for cam in camInstances:
        camera = cam.getCameraParam()
        print("intrinsicParam : " , camera["intrinsicParam"])
        print("extrinsicRotation : ", camera["extrinsicRotation"])
        print("extrinsicTranslation : ",camera["extrinsicTranslation"])
        print("distortionParam : ", camera["distortionParam"])
        print("Cam ID", camera["serialNo"])
        print("Camera Type", camera["cameraType"])

def createDatabase(customFeatures):
    """
    Create databse structure for storage of triangulated 3D features

    :return:
    """

    defaultDataSeries = {"frame": 0}
    for feature in customFeatures:
        defaultDataSeries[ str(feature) + "_x"] = 0
        defaultDataSeries[str(feature) + "_y"] = 0
        defaultDataSeries[str(feature) + "_z"] = 0

    dataFrame = pd.DataFrame(columns=list(defaultDataSeries))

    return defaultDataSeries, dataFrame

def avgList(listOfFeatures):
    """
    Give the avg from the list
    :param listOfFeatures : List of features
    :return: avg out values
    """
    avg = 0
    noOfZeroes = listOfFeatures.count(0)
    length = len(listOfFeatures) - noOfZeroes
    if length != 0:
        avg = sum(listOfFeatures)/length

    return avg


def getFinalPoints(dataFrame3DFeatures, features):
    """
    Get final points from the triangulated annotations
    :param dataFrame3DFeatures: Dataframe of features
    :param features: name of custom features
    :return: dict of normalised points
    
    NEW! By Alex Chan
    Auto reject points that are more than a threshold away from mean
    """
    dict = {}
    for feature in features :
        # Get coordinates of the features
        xFeatures = dataFrame3DFeatures[feature+ "_x"].tolist()
        yFeatures = dataFrame3DFeatures[feature + "_y"].tolist()
        zFeatures = dataFrame3DFeatures[feature + "_z"].tolist()

        #remove all 0s
        xFeatures = [i for i in xFeatures if i != 0]
        yFeatures = [i for i in yFeatures if i != 0]
        zFeatures = [i for i in zFeatures if i != 0]


        # Average all coordinates
        xAvg = np.nanmean(xFeatures)
        yAvg = np.nanmean(yFeatures)
        zAvg = np.nanmean(zFeatures)

        #initial error values:
        Points = np.array([xFeatures,yFeatures,zFeatures])
        DiffMean = np.sqrt((Points[0,:] - xAvg)**2 + (Points[1,:] - yAvg)**2 + (Points[2,:] - zAvg)**2)


        #auto reject points based on threshold of 5 vicon units
        #Get diffs:
        while True:
            # print(DiffMean)
            # import ipdb; ipdb.set_trace()

            #if there are points with >5 units off
            if len(np.where(DiffMean>5)[0]) != 0:
                Points = np.delete(Points,np.argmax(DiffMean),1)

                xAvg = np.nanmean(Points[0,:])
                yAvg = np.nanmean(Points[1,:])
                zAvg = np.nanmean(Points[2,:])

                DiffMean = np.sqrt((Points[0,:] - xAvg)**2 + (Points[1,:] - yAvg)**2 + (Points[2,:] - zAvg)**2)
            else:
                break

        # create dict entry for avg point
        dict[feature] = [xAvg, yAvg, zAvg]

    # import ipdb; ipdb.set_trace()
    return dict


def compute2DError(projectedPoints, originalAnnotations):
    """
    Compute 2D error between the annotated point and the reprojected 2D point after triangulation
    :param dict : projectedPoints: projection of all triangulated points on image space
    :param dict : triangulatedPoints: selected 2D annotation
    :return: dict : 2D error dictionary
    """
    errorDict = {}
    for feature in originalAnnotations:
        poin2D = originalAnnotations[feature]
        # Check if the required feature is annotated or not, if not we skip that feature
        if feature in projectedPoints.keys() :
            projectedPoint2D = projectedPoints[feature]
        else:
            continue

        error = np.sqrt( (poin2D[0]-projectedPoint2D[0]) * (poin2D[0]-projectedPoint2D[0]) +
                         (poin2D[1] - projectedPoint2D[1]) * (poin2D[1] - projectedPoint2D[1]) )
        errorDict[feature] = error

    return errorDict

def computeMean2DError(projectedPoints, originalAnnotations):
    """
    Author: Alex Chan
    Extended compute 2D error above to remove cases where points wasnt in original annotation (0,0)
    And computes mean error instead of error for each point
    ######

    Compute 2D error between the annotated point and the reprojected 2D point after triangulation
    :param dict : projectedPoints: projection of all triangulated points on image space
    :param dict : triangulatedPoints: selected 2D annotation
    :return: dict : 2D error dictionary
    """
    # import ipdb; ipdb.set_trace()
    errorDict = {}
    for feature in originalAnnotations:
        poin2D = originalAnnotations[feature]
        # Check if the required feature is annotated or not, if not we skip that feature
        if feature in projectedPoints.keys() and poin2D != [0,0]:
            projectedPoint2D = projectedPoints[feature]
        else:
            continue

        error = np.sqrt( (poin2D[0]-projectedPoint2D[0]) * (poin2D[0]-projectedPoint2D[0]) +
                (poin2D[1] - projectedPoint2D[1]) * (poin2D[1] - projectedPoint2D[1]) )
        errorDict[feature] = error

    AvgError = np.mean(list(errorDict.values()))
    return AvgError

def createAdvancedFeatures(features,subjects):
    """
    Author: Alex Chan
    Create longer list of all features for each individual pigeon

    """
    featureList = []

    for subject in subjects:
        for feature in features:
            featureList.append(str(subject) + "_"+ str(feature))

    return featureList

def CheckSubjectFile(subjects, directoryName):
    """
    Author: Alex Chan
    Checks if a given subject file exists, then ask user for feedback
    outputs list of subjects that the user wants to re-triangulate
    Params:
    :subject: list of subjects in the trial
    :directoryName: name of base directory of trial   
    """

    FinalSubjects = []
    ReusedSubjects = []
    for subject in subjects:
        SubjectFeatPath = "%s/%s.customFeatures.txt"%(directoryName, subject)
        if os.path.exists(SubjectFeatPath): #File already exists
            Feedback = input("Custom Feature File for Subject %s is found! \n Do you want to use pre-existing definition? \n Type 'Y' for yes and Type 'N' for no:"%subject)
            while Feedback != "Y" and Feedback !="N":
                Feedback = input("Input Incorrect! Please try again. \n Custom Feature File for Subject %s is found! \n Do you want to use pre-existing definition? \n Type 'Y' for yes and Type 'N' for no:"%subject)

            if Feedback == "N":#wont use old triangulation
                FinalSubjects.append(subject)
            elif Feedback == "Y":
                #reuse dict
                ReusedSubjects.append(subject)
            else:
                raise Exception("Bug in input reading")
        else: #file not found, need to triangulate
            FinalSubjects.append(subject)

    return FinalSubjects, ReusedSubjects

def FindClosestFrame(SyncArray, Frame):
    """
    Author: Alex Chan

    Given a sony frame and a SyncArray, find closest frame in vicon
    
    Will throw error if frame is before first flash
    """
    # import ipdb;ipdb.set_trace()
    #Index of frame where it is last synced
    #if minus the current frame, smaller than 1 (i.e any negative number and 0)
    index = np.where((SyncArray[0]-Frame) <1)[0].argmax()
    # index=4
    #find proportion of how far the frame is within the gap
    Prop = (Frame - SyncArray[0][index])/(SyncArray[0][index+1]- SyncArray[0][index])

    #Calculate corresponding vicon frame from the proportion
    ViconFrame = SyncArray[1][index]+(Prop*(SyncArray[1][index+1] - SyncArray[1][index]))

    return ViconFrame




def createTriangulatedFeatures(projectSettings):
    """
    Read the settings from the given file and read
    :param settingsDict: dict
    :return: None
    """
    # import ipdb;ipdb.set_trace()
    settingsDict = projectSettings.settingsDict
    # directoryName = settingsDict["rootDirectory"]
    sessionName = settingsDict["session"]
    objectsToTrack = settingsDict["objectsToTrack"] #objects to track per subject
    subjects = settingsDict["Subjects"]
    dataDir = settingsDict["DataDirectory"]


    # Part 1 : Load system information from calibration file i.e. calib, video files, .csv files and so on.
    SystemData = system.SystemInit(projectSettings)
    custom = True

    CoordData = SystemData.CoordData
    viconObjects = SystemData.viconObjects
    
    # Create video objects
    videoFiles = SystemData.sessionVideoFiles  # __getattribute__("sessionVideoFiles")
    viconCamObjects = SystemData.viconCameraObjects
    videoObjects = SystemData.viconVideoObjects
    imageObjects = SystemData.viconImageObjects
    # import ipdb; ipdb.set_trace()

    # Define window names
    # windowNames = []
    maxFrameNo = []
    for video in videoObjects:
        # windowNames.append(video.windowName)
        # cv.namedWindow(video.windowName, cv.WINDOW_NORMAL)
        maxFrameNo.append(video.totalFrameCount)
        
    #Only 1 window required now, with subject wise update
    cv.namedWindow("ReviewWindow", cv.WINDOW_NORMAL)

    
    # import ipdb; ipdb.set_trace()
    # Part 2 : Preapre files and features for triangultion of annotated features
    # Initialise class to read image annotations

    annotationFiles = SystemData.generate2DAnnotationFileNames()
    featureFileName = os.path.join(settingsDict["customFeatureFile"])
    outputFeatureFileName =  os.path.join(dataDir, settingsDict["customFeatureFile3D"])
    customFeatures = fileOp.readFeaturesFromFile(featureFileName)
    AllCustomFeatures = createAdvancedFeatures(customFeatures,subjects)

    imageAnnotationReaderObjects = []
    commonAnnotatedFrames = []
    # import ipdb;ipdb.set_trace()
    for annotationImageID in annotationFiles:
        # import ipdb;ipdb.set_trace()
        annotationObject = fileOp.ImageAnnotationDatabaseReader(annotationFiles[annotationImageID], AllCustomFeatures)
        commonAnnotatedFrames.append(annotationObject.data["frame"].tolist() )
        imageAnnotationReaderObjects.append(annotationObject)


    # Check if all videos have same number of frames and reassigns the variable to one single value
    if maxFrameNo.count(maxFrameNo[0]) == len(maxFrameNo) or custom:
        maxFrameNo = maxFrameNo[0]
    else:
        raise ValueError("Videos do not have same frame count.")

    # Create data frame for the triangulated features
    defaultTraingulatedFeatureDict3D, dataFrame3DFeatures = createDatabase(AllCustomFeatures)
    triagulatedPointsDatabasePath =  os.path.join(dataDir, settingsDict["dataFile3D"])


    # Create a triangulator class which would compute traingulated points for the given cameras
    stereoTriangulator = stereo.StereoTrinagulator(viconCamObjects, imageObjects)

    ####Looks for subjectFile, ask user if want to reuse feature file or no
    # import ipdb; ipdb.set_trace()
    FinalSubjects,ReusedSubjects = CheckSubjectFile(subjects, dataDir)

    if len(FinalSubjects)>0: #Added this, if all reused subjects, just skip everything
        # Part 3 : Process frame wise information and triangulate the features given in the annotation file
        # import ipdb;ipdb.set_trace()
        for frameNo in commonAnnotatedFrames[0]:
            # import ipdb; ipdb.set_trace()
            #Clear features stored in image class and camera object class, since they change per frame
            for j in range(len(viconCamObjects)):
                viconCamObjects[j].clearFeatures()
                imageObjects[j].clearFeatures()

            for j in range(len(viconObjects)):
                viconObjects[j].removeSelectedFeatures(customFeatures)

            # Go through the image annotation database and get features of image point
            featureDicts = []
            for i in range(len(imageAnnotationReaderObjects)):
                    featureDict = imageAnnotationReaderObjects[i].getDataForVideoFrame(frameNo)
                    imageObjects[i].setFeatures(featureDict)
                    featureDicts.append(featureDict)
            # import ipdb; ipdb.set_trace()
            # Traingulate points
            
            triangulatedDict = stereoTriangulator.TriangulatePointsNViewBA(custom)
            featureDictViconSpace = triangulatedDict #custom workaround to directly triangulate from pose of both cameras

            if len(triangulatedDict) == 0:
                print(" No traingulated features for frame no : ", frameNo)
                continue

            # Set features in cam space and transform features to vicon space

            print("feature dict" , featureDictViconSpace)


            # Get rotation and translation information of tracking objects for given frame
                #for custom sony: just use cam 1, should all be synced at annotation step anyways
            # RealFrameNo = frameNo - viconCamObjects[0].FrameDiff #for custom sony cameras, theres time lag, correct for it
            # frameNo = frameNo + 163
            ViconFrameNo = FindClosestFrame(viconCamObjects[0].SyncArray, frameNo)
            # import ipdb;ipdb.set_trace()
            coordinateDict = CoordData.getCoordDataForViconFrame(round(ViconFrameNo))


            traingulatedFeatureDict3D = defaultTraingulatedFeatureDict3D.copy()
            traingulatedFeatureDict3D["frame"] = frameNo
            # import ipdb; ipdb.set_trace()
            # print(frameNo)
            #get triangulated points in object space:

            for object,j in zip(objectsToTrack,range(len(viconObjects)) ):
                viconObjects[j].Status = "Good" #newly added, record if object is not tracked in vicon

                ##convert coordinate to rotation translation
                ObjectCoordDict = fileOp.get3DDictofObject(coordinateDict, object)
                # import ipdb; ipdb.set_trace()
                #for rare exception where frame is not well tracked, and object is NA
                FlatList = [item for sublist in ObjectCoordDict.values() for item in sublist]
                if np.isnan(FlatList).any():
                    # put identity matrix for rot trans, then clear features
                    # import ipdb; ipdb.set_trace()
                    viconObjects[j].Status = "Bad"
                    viconObjects[j].setTransformationParameters(np.identity(3),np.zeros((3,1)))
                    continue

                Rotation,Translation = viconObjects[j].GetRotTransFromViconSpace(ObjectCoordDict)
                

                # Rotation = list(tf.rotMatrixToQuat(Rotation))
                # If transformation is valid then transfer features into object space
                # if transformationParamDict[object+"_validity"]:
                    # If transformation is valid it is stored
                viconObjects[j].setTransformationParameters(Rotation,Translation)

                #Get FeatureDictViconSpace for this specific object
                FeatureNames = list(featureDictViconSpace.keys())
                MatchedNames = [x for x in FeatureNames if x.startswith(object)]
                featureDictViconSpaceObject = {key:value for key,value in featureDictViconSpace.items() if key in MatchedNames }
                # import ipdb; ipdb.set_trace()
                ####If no custom features triangulated for given object,just continue
                if len(featureDictViconSpaceObject) == 0:
                    # viconObjects[j].setFeatures(objectSpecificFeatureDict)
                    # print("Object Points", viconObjects[j].__getattribute__("featureDict"))
                    continue

                featureDictObjectSpace = viconObjects[j].transferFeaturesToObjectSpaceSimple(featureDictViconSpaceObject)
                print("Features Object Space: ", featureDictObjectSpace)
                # import ipdb; ipdb.set_trace()


                objectSpecificFeatureDict = {}
                # For each feature transferred to an object space, we check if it belongs a tracking object object
                for feature in featureDictObjectSpace:
                    if viconObjects[j].object in feature:  # object name must be in feature name
                        point3D = featureDictObjectSpace[feature]
                        objectSpecificFeatureDict[feature] = point3D
                        traingulatedFeatureDict3D[feature + '_x'] = point3D[0]
                        traingulatedFeatureDict3D[feature + '_y'] = point3D[1]
                        traingulatedFeatureDict3D[feature + '_z'] = point3D[2]

                # Set features for the object.
                viconObjects[j].setFeatures(objectSpecificFeatureDict)
                print("Object Points", viconObjects[j].__getattribute__("featureDict"))
            # import ipdb; ipdb.set_trace()
                # else:
                #     print("Invalid transformation for object : ", object, "for frame : ", frameNo )

            # Transfer all the points from object space to vicon space (Feature points + Newly traingulated points
            # import ipdb; ipdb.set_trace()

            transferredFeatureDictViconSpace = {}
            for i in range(len(viconObjects)):                
                if viconObjects[i].Status =="Bad":
                    #cleared features above, skip
                    # import ipdb; ipdb.set_trace()
                    continue

                dict = viconObjects[i].transferFeaturesToViconSpaceSimple()
                transferredFeatureDictViconSpace.update(dict)

            # Now transfer all points from VICON object space to Camera space
            for j in range(len(viconCamObjects)): # Loop through camera objects and find image projections
                # Dictionary operations
                featureDictCamSpace = viconCamObjects[j].transferFeaturesToObjectSpace(transferredFeatureDictViconSpace,custom)
                #print("Features vicon -> camera space : ", featureDictCamSpace)
                viconCamObjects[j].setFeatures(featureDictCamSpace)
                imageFeaturesDict = imageObjects[j].projectFeaturesFromCamSpaceToImageSpace(featureDictCamSpace)
                imageObjects[j].setFeatures(imageFeaturesDict)
                # print("image Projections Features: ", imageObjects[j].__getattribute__("featureDict"))
                # import ipdb; ipdb.set_trace()
                error = compute2DError(imageFeaturesDict, featureDicts[j])
                MeanError = computeMean2DError(imageFeaturesDict, featureDicts[j])
                print("Mean Error for cam ", j, " is", error)
                # import ipdb;ipdb.set_trace()

            #### Looping through each subject and accepting and rejecting per subject
            for subject in FinalSubjects:
                temp = [k for k,v in viconCamObjects[0].featureDict.items()if k.startswith(subject)]
                if len(temp) == 0:
                    #subject not plotted, skip
                    continue  


                CroppedWindows = []
                for j in range(len(videoObjects)):

                    if custom:
                        realFrame = frameNo- viconCamObjects[j].FrameDiff #for sony cameras, fix time lag
                    else:
                        realFrame = frameNo
                    
                    image = videoObjects[j].getFrame(realFrame)
                    # import ipdb;ipdb.set_trace()
                    if image is not None:
                        imageObjects[j].drawFeatures(image, 1) #draw all features
                        clone = image.copy()
                        BBox = imageObjects[j].computeSubjectBoundingBox(subject)
                        BBoxPoints = list(BBox.values())
                        roiImage = clone[int(BBoxPoints[0][1]):int(BBoxPoints[1][1]), int(BBoxPoints[0][0]):int(BBoxPoints[1][0]),:]
                        # import ipdb; ipdb.set_trace()
                        #check if empty
                        if not roiImage.any():
                            resized = np.zeros((200,200,3))
                        else:
                            resized = cv.resize(roiImage, (200,200), interpolation = cv.INTER_AREA)
                        CroppedWindows.append(resized)                    
                
                # import ipdb; ipdb.set_trace()

                SplitIndex = [[k, k+1] for k in range(0,len(CroppedWindows),2)]
                FinalImage = np.concatenate([np.concatenate([CroppedWindows[x[0]],CroppedWindows[x[1]]],axis=1 ) for x in SplitIndex], axis=0)
                
                            #add black border to image
                FinalImageBorder = np.zeros((450,400,3), dtype="uint8")
                FinalImageBorder[50:450,:,:] = FinalImage

                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(FinalImageBorder, "frameNo : " + str(frameNo), (10,10), font, 0.4,
                        (255, 255, 255), 1, cv.LINE_AA)
                cv.putText(FinalImageBorder, "Press : A-ccept, R-eject, Q-uit, N-ext", (10,25), font, 0.4,
                        (255, 255, 255), 1, cv.LINE_AA)
                cv.putText(FinalImageBorder, "Subject : " + str(subject), (10,40), font, 0.4,
                        (255, 255, 255), 1, cv.LINE_AA)

                cv.imshow("ReviewWindow",FinalImageBorder)
                k = cv.waitKey(0)

                if k == ord('q'):
                    cv.destroyAllWindows()
                    break

                if k == ord('a'):
                    print("Accept Annotation")
                    ##### only update 3D csv for given subject for given frame
                    AcceptedFeatures = {k:v for k,v in traingulatedFeatureDict3D.items() if k.startswith(subject) or k=='frame'}
                    
                    ##########SET BY FRAME + COLUMN NAME #####
                    ###### if NO FRAME FOUND, APPEND ######

                    if AcceptedFeatures["frame"] in dataFrame3DFeatures["frame"].tolist():
                        rowIndex = np.where(dataFrame3DFeatures["frame"]==AcceptedFeatures["frame"])[0][0]
                        #row for frame already exist, fill in cells
                        for key, value in AcceptedFeatures.items():
                            dataFrame3DFeatures.loc[rowIndex, key] = value
                    else:
                        dataFrame3DFeatures = dataFrame3DFeatures.append(AcceptedFeatures, ignore_index=True)

                    dataFrame3DFeatures.to_csv(triagulatedPointsDatabasePath, index=False)  # Save each entry in the file
                    #traingulatedFeatureDict3D = {}
                if k == ord('r'):
                    print("Reject Annotation")

                if k == ord('n'):
                    continue
        cv.destroyAllWindows()

        # After triangulation go through all triangulated points and create a single feature.
        finalPointDict = getFinalPoints(dataFrame3DFeatures, AllCustomFeatures)
        print("Final Point Dict: ", finalPointDict)
        # import ipdb; ipdb.set_trace()
        
    else:
        finalPointDict = {}

    ##for the subjects that is already triangulated, re-read the data and update
    for Resubject in ReusedSubjects:
        SubjectFeatPath = "%s/%s.customFeatures.txt"%(dataDir, Resubject)
        #Update dict
        SubjectFeat = fileOp.readFeaturePointsFromTextFile(SubjectFeatPath)
        for key,value in SubjectFeat.items():
            finalPointDict[key] = value

    fileOp.writeFeaturePointsToFile(outputFeatureFileName, finalPointDict)
    
    #Write txt output for each individual subject
    for subject in subjects:
        outputPath = "%s/%s.customFeatures.txt"%(dataDir, subject)
        SubjectfeatureDict = {k:v for k,v in finalPointDict.items() if k.startswith(subject)}
        fileOp.writeFeaturePointsToFileSubject(outputPath, SubjectfeatureDict,subject)


def main(settingsFile):
    projectSettings = settingsGenerator.xmlSettingsParser(settingsFile)
    createTriangulatedFeatures(projectSettings)



if __name__ == '__main__':
    # In this section we pass the arguments required for the main function to perform any operation.
    # Mostly it is the system information for calibration and the location of the VICON file

    settingsFile = "D:\\BirdTrackingProject\\20190618_PigeonPostureDataset\\settings_session02.xml"
    main(settingsFile)


