# This code is used in case of simple math operations
from dis import dis

import cv2 as cv
import pandas as pd
import numpy as np
import h5py


def pointDistance3D(dict,dictNexus):
    """
    Find distance between two points
    :param dict: dict of 3D points
    :param dictNexus: dict of 3D points
    :return: dict with diff between features
    """
    distDict = {}

    distDict = dict.copy()
    distDict.update(dictNexus)
    for feature in distDict:
        distDict[feature] = np.NaN

    for point in dict:
        # If feature is in both files
        if point in dictNexus:
            pTracker = dict[point]
            pNexus = dictNexus[point]
            distance = findDistance3D(pTracker,pNexus)
            distDict[point] = distance
        else:
            distDict[point] = np.NaN
            print("Point does not exist")

    return distDict

def findDistance3D(point1, point2):
    """
    Compute distance between two points (as list)
    :param point1: list points 1
    :param point2: list points 2
    :return: distance
    """
    assert (type(point1) == list and type(point2) == list)," Unexpected input type, a list is required"
    xDist = point1[0] - point2[0]
    yDist = point1[1] - point2[1]
    zDist = point1[2] - point2[2]

    dist = np.sqrt((xDist*xDist) + (yDist*yDist) + (zDist*zDist))
    return dist

def filterData(data):
    """
    Removes points with no data from 3xN matrix.
    :param data: 3xN point array.
    :return: (array) modified 3xM matrix , (list) column indexes of non-zero points (len M).
    """
    #Data = 3XN
    centroid_over_coordinates = np.mean(data,axis = 0)
    indexOfNonZeroElements = np.where(centroid_over_coordinates != 0)[0]
    modifiedData = data[:,indexOfNonZeroElements]
    # output = MxN , Index of non-zero data point
    return modifiedData, indexOfNonZeroElements

def filterDataForPoint(data, point):
    """
    Removes points with no data from 3xN matrix.
    :param data: 3xN point array.
    :return: (array) modified 3xM matrix , (list) column indexes of non-zero points (len M).
    """
    #Data = 3XN
    assert( data.shape[0] == 3), "Expected array in shape 3xN"
    assert (point.shape == (3,1) ),"Expected array in shane 3xN"

    # Create a one dimensional array with distance of all points with the query point
    distArray = distFromPoint(data, point)
    listOfDistances = distArray[0,:].tolist()
    indexOfNonMatchingPoints = np.where(listOfDistances != 0)[0]
    indexOfMatchingPoints = np.where(listOfDistances == 0)[0]
    modifiedData = data[:,indexOfNonMatchingPoints]
    # output = MxN , Index of non-zero data point
    return modifiedData,indexOfMatchingPoints,indexOfNonMatchingPoints


def distFromPoint(data_points, centroid_point):
    """
    Copute distances of each point in the data with given centroid point
    :param data_points: 3xN array
    :param centroid_point: 3x1 array
    :return 1xN array of distances of each points in data_points with centroid_point
    """
    #find distance of each point
    rows, cols = data_points.shape
    assert(rows == 3),"Shape of the given data is required to be in 3xN or 2xN, with N > 1"
    centroid_point= centroid_point.reshape(3,1)
    centroid_tiles = np.tile(centroid_point,(1,cols))
    dist = np.sqrt(np.sum(np.square(data_points - centroid_tiles),axis=0))
    return dist

def createDistanceMatrix(data_points):
    """
    compute distance of each point with other points matrix 3xN is required
    :param data_points: 3xN array
    :return: NxN array of distance between each points
    """
    dist_matrix = np.zeros((data_points.shape[1],data_points.shape[1]))
    for i in range(data_points.shape[1]):
        point = data_points[:,i]
        dist = distFromPoint(data_points,point)
        dist_matrix[i,:] = dist
    return dist_matrix

def main():
    print("Math Operations")
    point1 = [1,2,3]
    point2 = [1.1,2.1,3.1]

    distance  = findDistance3D(point1,point2)
    print("Distance : ", distance)

    dict1 = {"f1" : point1, "f2": point2} #, "f3": point1}
    dict2 = {"f1": point2, "f2": point2} #, "f4": point1}
    distDict = pointDistance3D(dict1,dict2)

    print("Dist Dict : ", distDict)

    point = np.zeros((3, 4))
    point[:, 1] = 1
    output, index = filterData(point)
    print(" point: ", point)
    print(" output: ", output)
    print(" index: ", index)


if __name__ == '__main__':
    main()