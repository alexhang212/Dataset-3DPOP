"""
Author: Alex Chan

Part of the auto-keypoint annotation pipeline, takes annotations and apply it accross whole 
trial to generate 3D keypoints ground truth
"""

from logging.handlers import RotatingFileHandler
from System import systemInit as system
from FileOperations import rwOperations as fileOp
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
from scipy.spatial.transform import Rotation


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


def Generate3DKeypoint(projectSettings):
    """
    Author:Alex Chan
    Takes annotation from annotation pipeline then applies it accross whole trial


    """
    #load required files
    # import ipdb;ipdb.set_trace()
    settingsDict = projectSettings.settingsDict
    AnnotDir = settingsDict["AnnotationDirectory"]
    dataDir = settingsDict["DataDirectory"]

    viconSystemData = system.SystemInit(projectSettings)


    outputFeatureFileName =  os.path.join(dataDir, settingsDict["customFeatureFile3D"])

    # viconObjects = viconSystemData.viconObjects

    ###Apply defined feature to whole trial, output csv with vicon 3D points
    TrialCoord = viconSystemData.CoordData

    #load all files and saved object definitions
    Features = []
    viconObjectsFinal = viconSystemData.loadVICONObjectsFromSubject()
    CustomFeaturesAll = fileOp.readFeaturePointsFromTextFile(outputFeatureFileName)

    for x in range(len(viconObjectsFinal)):
        BaseFeatures = viconObjectsFinal[x].featureDict
        CustomFeatures = {key:val for key,val in CustomFeaturesAll.items() if key.startswith(viconObjectsFinal[x].name) }
        
        viconObjectsFinal[x].setFeatures({**BaseFeatures, **CustomFeatures})

        Features.extend(list(viconObjectsFinal[x].featureDict.keys()))

    # import ipdb; ipdb.set_trace()

    FullDFDict = {} #Not using pandas append, append to dictionary instead

    print("Applying custom features to the whole trial")
    for i in tqdm(range(len(TrialCoord.data))):
    # for i in tqdm(range(12000)):
        # import ipdb; ipdb.set_trace()
        i = TrialCoord.data["Frame"][i]
        TrialFeatureCoordDict, _ = createDatabase(Features)

        coordinateDict = TrialCoord.getCoordDataForViconFrame(i)
        transferredFeatureDictViconSpace = {}
        for object in viconObjectsFinal:
            ObjectCoordDict = fileOp.get3DDictofObject(coordinateDict, object.name)
            Rotation,Translation = object.GetRotTransFromViconSpace(ObjectCoordDict)

            if Rotation is False:
                # import ipdb; ipdb.set_trace()
                #Cannot compute rotation/ translation, just return NA for all values
                Outdict = object.featureDict.fromkeys(object.featureDict.keys(),[float('nan'),float('nan'),float('nan')])
                transferredFeatureDictViconSpace.update(Outdict)
            else:
                object.setTransformationParameters(Rotation,Translation)
                Outdict = object.transferFeaturesToViconSpaceSimple()
                transferredFeatureDictViconSpace.update(Outdict)
            
        ##Generate dictionary with split coordinates
        for feature in transferredFeatureDictViconSpace:
            point3D = transferredFeatureDictViconSpace[feature]
            TrialFeatureCoordDict[feature + '_x'] = point3D[0]
            TrialFeatureCoordDict[feature + '_y'] = point3D[1]
            TrialFeatureCoordDict[feature + '_z'] = point3D[2]
        TrialFeatureCoordDict["frame"] = i

        # import ipdb; ipdb.set_trace()

        #Save Data:
        FullDFDict.update({str(i):TrialFeatureCoordDict})


    TrialFeatureCoordDF = pd.DataFrame.from_dict(FullDFDict, orient="index")
    #Save csv
    TrialFeatureCoordPath =  os.path.join(AnnotDir, settingsDict["FinalFeatureCSV"])
    TrialFeatureCoordDF.to_csv(TrialFeatureCoordPath, index=False)
