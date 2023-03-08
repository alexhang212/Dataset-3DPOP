
#

from FileOperations import rwOperations
import glob
import os
import h5py
import pandas as pd


def processData(dataFrame, filterBboxData = True):
    """
    Process data to remove all the non zero 2D coordinates from the given dataframe
    :param dataFrame:
    :return:
    """
    print("Process")
    featureList = dataFrame.columns.tolist()
    print("Feature list : ", featureList)
    if filterBboxData:
        featureList = [x for x in featureList if "Corner" not in x]
        print("Filtered list : ", featureList)

    # Remove all the data that is erroneous or not valid in form of image coordinates.
    for feature in featureList:
        if "2d" in feature and "x" in feature :
            print(" Cleaning for feature : {0}, dataframe shape {1} ".format(feature,dataFrame.shape) )
            dataFrame = dataFrame[(dataFrame[feature] > 0) & (dataFrame[feature] < 1920)]
            print("Data Shape after cleaning the erroneous bounding boxes ", dataFrame.shape)

        elif "2d" in feature and "y" in feature :
            print(" Cleaning for feature : {0}, dataframe shape {1} ".format(feature, dataFrame.shape))
            dataFrame = dataFrame[(dataFrame[feature] > 0) & (dataFrame[feature] < 1080)]
            print("Data Shape after cleaning the erroneous bounding boxes ", dataFrame.shape)
        else:
            print(" Nothing to be done for feature : {0} ".format(feature))

    return dataFrame

def divideData(dataFrame):

    columnData = dataFrame.columns.tolist()

    features2D = [x for x in columnData if "2d" in x and "Corner" not in x]
    features3D = [x for x in columnData if "3d" in x]
    print("Features for 2D {0} \n for 3D {1}".format(features2D,features3D))
    dataFrame2D = dataFrame[features2D]
    dataFrame3D = dataFrame[features3D]

    return dataFrame2D, dataFrame3D

def replace3DFeatures(dataFrame1, dataFrame2):
    """
    Replace 3D points of one frame with another
    :param dataFrame1:
    :param dataFrame2:
    :return:
    """
    columnData = dataFrame1.columns.tolist()
    features3D = [x for x in columnData if "3d" in x]
    dataFrame1[features3D] = dataFrame2[features3D]


if __name__ == '__main__' :
    # Parent directory for the data
    dir = "data/*"
    print("File exists" , os.path.exists(dir))

    files = glob.glob(dir)
    cameras = ["2118670","2119571"]
    sessions = ["session","session01","session02"]

    training_data = []
    test_data = []

    subjectId = ["1","9"]
    action = ["walking"]
    seq = ["session","session01","session02"]
    files = [x for x in files if x.endswith(".csv")]

    print(files)
    fileDict = {}

    # Divide the files based on the sessions
    for sess in sessions:
        file = [x for x in files if sess+"." in x]
        fileDict[sess] = file

    # Now read the files and combine the dataset
    for sess in sessions:
        files = fileDict[sess]
        # Read the data base from both the files
        originCameraIndex = 0
        dataFrameDict = {}
        # Store the 3D data frame of each file in a dict
        for file in files:
            featureList = rwOperations.readFeaturesFromFile("D:\\BirdTrackingProject\\20190618_PigeonPostureDataset\\20190618_PigeonPostureDataset_session02.customFeatures.txt")
            dataBaseObject = rwOperations.annotationDatabase(file, featureList)
            if cameras[0] in file:
                originCameraIndex = files.index(file)
            dataFrameDict[file] = dataBaseObject.dataBase

        for i in range(len(files)):
            fileName = files[i].split(".csv")[0]
            # If the current file does not belong to origin camera, replace the 3D data
            if originCameraIndex != i:
                print(" Data print before : ", dataFrameDict[files[i]].iloc[0])
                replace3DFeatures(dataFrameDict[files[i]], dataFrameDict[files[originCameraIndex]])
                print(" Data print after : ", dataFrameDict[files[i]].iloc[0])
            # Process the data frame to have no holes
            dataBase = processData(dataFrameDict[files[i]])
            dataBase2D, dataBase3D = divideData(dataBase)
            dataBase2D.to_hdf(fileName+ ".h5","2DPositions")
            dataBase3D.to_hdf(fileName+ ".h5","3DPositions")


    dict3D = {}
    dict2D = {}
    i = 0
    files = glob.glob(dir)
    files = [x for x in files if x.endswith(".h5")]

    for file in files :
        fileName = file.split(".h5")[0]
        frame = pd.read_hdf(fileName + ".h5", "3DPositions")
        print("3D position {0}: \n Shape : {1}".format(fileName + ".h5", frame.as_matrix().shape))
        #print("1 Row : ", frame.iloc[0])

        frame = pd.read_hdf(fileName + ".h5", "2DPositions")
        print("2D Position {0}: \n Shape : {1}".format(fileName + ".h5", frame.as_matrix().shape))
        #print("1 Row : ", frame.iloc[0])
        i = i + 1

        # dataMatrix = dataBase.as_matrix()
        # dataMix = dataBase.to_hdf(fileName+".h5", "data", mode = "w")
        #
        # frameNo = dataMatrix[:,0]
        # features2D = dataMatrix[:,feat2DIndexStart:feat2DIndexEnd]
        #
        # features3D = dataMatrix[:,feat3DIndexStart:feat3dIndexEnd]
        # bBoxFeat = dataMatrix[:,bBoxIndexStart:]
        #
        # print(dataMatrix.shape)
        # mat = features2D[0,:]
        # print(mat.shape)
        # print("File Name : ", file )
        # print(" Famre No {0}: \n 2D Features {1}: \n 3D Features{2} \n Bouding box {3}:".format(frameNo[1],features2D[1,:],features3D[1,:],bBoxFeat[1,:]))
        # print(" Shape {0}: \n 2D Features Shape {1}: \n 3D Features shape {2} \n Bouding box shape {3}:".format(frameNo.shape, features2D.shape,
        #                                                                                         features3D.shape, bBoxFeat.shape))