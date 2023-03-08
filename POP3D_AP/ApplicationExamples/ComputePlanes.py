"""

Compute planes of head and backpack. Here, for the purpose of calculating pose.
"""

from System import systemInit as system
import pandas as pd
import os
import numpy as np
from tqdm import tqdm


def Extract3DPoints(viconObject,rowData):
    """Extract 3 points to calculate plane of head or backpack"""

    OutDict = {}


    if viconObject.name.endswith("hd"):
        OutDict["origin"] = np.array([rowData["%s_beak_x"%viconObject.name],rowData["%s_beak_y"%viconObject.name],rowData["%s_beak_z"%viconObject.name]])
        OutDict["pt1"] = np.array([rowData["%s_leftEye_x"%viconObject.name],rowData["%s_leftEye_y"%viconObject.name],rowData["%s_leftEye_z"%viconObject.name]])
        OutDict["pt2"] = np.array([rowData["%s_rightEye_x"%viconObject.name],rowData["%s_rightEye_y"%viconObject.name],rowData["%s_rightEye_z"%viconObject.name]])

    elif viconObject.name.endswith("bp"):
        OutDict["origin"] = np.array([rowData["%s_tail_x"%viconObject.name],rowData["%s_tail_y"%viconObject.name],rowData["%s_tail_z"%viconObject.name]])
        OutDict["pt1"] = np.array([rowData["%s_leftShoulder_x"%viconObject.name],rowData["%s_leftShoulder_y"%viconObject.name],rowData["%s_leftShoulder_z"%viconObject.name]])
        OutDict["pt2"] = np.array([rowData["%s_rightShoulder_x"%viconObject.name],rowData["%s_rightShoulder_y"%viconObject.name],rowData["%s_rightShoulder_z"%viconObject.name]])

    else:
        print("Something wrong")

    return OutDict

def createAngleDatabase(customObjects):
    """
    Create databse structure for euler angles

    :return:
    """
    # objects = ["hd","bp"]
    defaultDataSeries = {"frame": 0}
    for feature in customObjects:
        defaultDataSeries[str(feature) + "_x"] = 0
        defaultDataSeries[str(feature) + "_y"] = 0
        defaultDataSeries[str(feature) + "_z"] = 0

    dataFrame = pd.DataFrame(columns=list(defaultDataSeries))

    return defaultDataSeries, dataFrame





def GetVectorNormalAngles(projectSettings):
    """ For each 3D frame, compute vector normal of head and back plane, then get angles"""

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

    TrialFeatureCoordPath =  os.path.join(directoryName, settingsDict["FinalFeatureCSV"])
    AngleSavePath =  os.path.join(directoryName, settingsDict["session"]+"_RadianAngles.csv")

    Trial3Ddf = pd.read_csv(TrialFeatureCoordPath)

    viconObjectsFinal = viconSystemData.loadVICONObjectsFromSubject()
    ObjectNames = []
    for x in range(len(viconObjectsFinal)):
        ObjectNames.append(viconObjectsFinal[x].name)

    FullDFDict = {}
    for i in tqdm(range(len(Trial3Ddf))):
        row = Trial3Ddf.iloc[i]
        TrialFeatureCoordDict, _ = createAngleDatabase(ObjectNames)
        FinalAngleDict = {}

        #for each object:
        for viconObject in viconObjectsFinal:
            # viconObject = viconObjectsFinal[0]
            outDict = {}
            
            #Extract 3 points for objects:
            PointDict = Extract3DPoints(viconObject, row)
            #Normalize to origin:
            # Org = PointDict["origin"] - PointDict["origin"]
            Norm1 =PointDict["pt1"] - PointDict["origin"]
            Norm2 = PointDict["pt2"] - PointDict["origin"]

            #Get 2 vectors
            VecNormal = np.cross(Norm1,Norm2)
            magnitute = np.sqrt(VecNormal[0]**2 +VecNormal[1]**2+VecNormal[2]**2)
            UnitVec = VecNormal/magnitute

            Angx = np.dot(UnitVec, np.array([1,0,0]))
            Angy = np.dot(UnitVec, np.array([0,1,0]))
            Angz = np.dot(UnitVec, np.array([0,0,1]))
            # import ipdb;ipdb.set_trace()
            outDict = {viconObject.name:[Angx,Angy,Angz]}
            FinalAngleDict.update(outDict)
        # import ipdb;ipdb.set_trace()
        for feature in FinalAngleDict:
            Angle = FinalAngleDict[feature]
            TrialFeatureCoordDict[feature + '_z'] = Angle[0]
            TrialFeatureCoordDict[feature + '_y'] = Angle[1]
            TrialFeatureCoordDict[feature + '_x'] = Angle[2]
        # import ipdb;ipdb.set_trace()
        # print(TrialFeatureCoordDict.keys())

        FullDFDict.update({str(i):TrialFeatureCoordDict})

    TrialFeatureCoordDF = pd.DataFrame.from_dict(FullDFDict, orient="index")
    TrialFeatureCoordDF.to_csv(AngleSavePath, index=False)





