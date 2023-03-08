""""
Author: Alex Chan
Reproject output from annotation sequence to 2D views for verification
"""

import sys
sys.path.append('./')

from FileOperations import settingsGenerator
from System import systemInit as system
import numpy as np
import cv2 as cv
from FileOperations import rwOperations
from POP3D_AP.System import SettingsGenerator
import os

def getFrameData(df, dataFrameNumber, FeatureList):
    """
    provides data for the query frame number
    :param dataFrameNumber: query frame
    :return: dict
    """
    dataFrameNumber = ((2)*dataFrameNumber)
    OutDict=dict.fromkeys(FeatureList)
    if not df.data[df.data["frame"] == dataFrameNumber].empty:
        dataList = df.data[df.data["frame"] == dataFrameNumber].values[0].tolist()
        subDataList = dataList[1:]

        if len(FeatureList)*3 != len(subDataList):
            print("Feature and column list of data do not meet")
            # return False

        for feature,index in zip(FeatureList,range(len(FeatureList))):
            xData = subDataList[3*index + 0]
            yData = subDataList[3*index + 1]
            zData = subDataList[3*index + 2]
            OutDict[feature] = [xData, yData, zData]

        return OutDict



def ReprojectResults(projectSettings, AllView = False, Cam = 1,SavetoFile=False, custom=False):
    """Read in final csv and reproject points to 2D"""
    settingsDict = projectSettings.settingsDict
    sessionName = settingsDict["session"]
    # FinalFeatureCoord = settingsDict["FinalFeatureCSV"]

    viconSystemData = system.VICONSystemInit(projectSettings,sessionName)
    
    #read video files
    videoFiles = viconSystemData.sessionVideoFiles  # __getattribute__("sessionVideoFiles")
    viconCamObjects = viconSystemData.viconCameraObjets
    videoObjects = viconSystemData.viconVideoObjects
    imageObjects = viconSystemData.viconImageObjects

    #Load final coordinate data output of all custom features
    if custom:
        df = rwOperations.MatlabCSVReader(os.path.join(settingsDict["rootDirectory"], settingsDict["FinalFeatureCSV"])) 
        FeatureNameList = df.data.columns.tolist()[1:]
    else:
        CoordDataObject = viconSystemData.loadViconCoord()
        FeatureNameList = CoordDataObject.data.columns.tolist()[1:]

    
    CustomFeatures = [FeatureNameList[x*3].strip("_x") for x in range(int(len(FeatureNameList)/3))]


    ##Show Video
    cv.namedWindow("temp", cv.WINDOW_NORMAL)

    frameNo = 50000
    #Save output video to file
    if SavetoFile:
        out = cv.VideoWriter("../MAAP3D/TempSamples/ViconFeature.mp4", cv.VideoWriter_fourcc(*'mp4v'), 30, (imageObjects[0].imageWidth,imageObjects[0].imageHeight))

    while(frameNo < videoObjects[0].totalFrameCount):
        images = []
        if AllView: #view on all 4 cameras at the same time for fun
            #But runs really slowly :(
            for j in range(len(videoObjects)):
                #transfer vicon object to camera object
                CoordDict = CoordDataObject.getFrameData(frameNo, CustomFeatures)


                featureDictCamSpace = viconCamObjects[j].transferFeaturesToObjectSpace(CoordDict)
                #print("Features vicon -> camera space : ", featureDictCamSpace)
                viconCamObjects[j].setFeatures(featureDictCamSpace)
                imageFeaturesDict = imageObjects[j].projectFeaturesFromCamSpaceToImageSpace(featureDictCamSpace)
                imageObjects[j].setFeatures(imageFeaturesDict)

                # image = videoObjects[j].s(frameNo)
                image = videoObjects[j].getFrame(frameNo)
                imageObjects[j].drawFeatures(image, 2)
                images.append(image)
            
            SplitIndex = [[k, k+1] for k in range(0,len(images),2)]

            FinalImage = np.concatenate([np.concatenate([images[x[0]],images[x[1]]],axis=1 ) for x in SplitIndex], axis=0)

        else:
            CamIndex = Cam-1
            if custom:
                # import ipdb; ipdb.set_trace()
                CoordDict = getFrameData(df, frameNo, CustomFeatures)
            else:
                # import ipdb ; ipdb.set_trace()
                CoordDict = CoordDataObject.getFrameData(frameNo, CustomFeatures)

            featureDictCamSpace = viconCamObjects[CamIndex].transferFeaturesToObjectSpace(CoordDict)
            #print("Features vicon -> camera space : ", featureDictCamSpace)
            viconCamObjects[CamIndex].setFeatures(featureDictCamSpace)
            imageFeaturesDict = imageObjects[CamIndex].projectFeaturesFromCamSpaceToImageSpace(featureDictCamSpace)
            imageObjects[CamIndex].setFeatures(imageFeaturesDict)

            FinalImage = videoObjects[CamIndex].getFrame(frameNo)

            imageObjects[CamIndex].drawFeatures(FinalImage, 1)

        cv.imshow("temp", FinalImage)
        if SavetoFile:
            out.write(FinalImage)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        frameNo += 1

        print(frameNo)
    cv.destroyAllWindows()
    if SavetoFile:
        out.release()




if __name__ == "__main__":
    SettingsFile =   "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/29062022Pigeon10/Vicon/Vicon_settings_training_29062022_08.xml"
    # SonyVideoName = "C0015"
    # TrackedSubjects = ["47_0107","54_0107","391_0107","452_0107","473_0107","484_0107","485_0107","486_0107","487_0107","705_0107"]
    # TrackedSubjects = ["47_0407","382_0407","389_0407","391_0407","452_0407","475_0407","497_0407","705_0407","706_0407","707_0407"]
    # TrackedSubjects = ["382_1307","391_1307","473_1307","475_1307","483_1307","483_1307","486_1307","486_1307","497_1307","706_1307","707_1307"]
    TrackedSubjects = []

    projectSettings = settingsGenerator.xmlSettingsParser(SettingsFile,TrackedSubjects)
    settings = projectSettings.settingsDict

    ReprojectResults(projectSettings, AllView = False, Cam = 2, SavetoFile=False,custom=True)