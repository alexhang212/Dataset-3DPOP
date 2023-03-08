# The file has purpose of reading the original VICON.csv export file and convert it to locally readable export file.
# This format shall be similar to the file generated
import os
import pandas as pd
import csv
import math
import numpy as np


def separateObjectName(name):
    separateColon = name.split(":")
    print(separateColon)
    spaseSeparatedWords = separateColon[0].split(" ")
    objectName = []
    for words in spaseSeparatedWords:
        if words != "Global" and words != "Angle":
            objectName.append(words)
    return objectName[0]


def getObjectNames(rowCentent):
    objectName = []
    noOfObjects = math.floor ( (len(rowCentent) - 2)/6)
    offSet = 2
    for i in range(noOfObjects):
        name = rowCentent[offSet + (i*6) ]
        objectName.append(separateObjectName(name))
    return objectName

def getObjectInfo(row):
    print("Get object related information")

def writeCSV(csvFile):
    """The function writes *.csv for pandas"""


def getRowData(rowData, objNames):
    """This function converts the row data in requied format for storing in the new .csv file """
    frameCount = 0
    objectData = []
    objects = []
    for i in range(len(objNames)):
        if int(rowData[1]) == 0:
            frame = int(rowData[0])
            startIndex = 2 + (i*6)
            stopIndex = startIndex + 6
            data = rowData[startIndex:stopIndex] # Get rotation and translation data from the data file
            if data[0]: # If the rotation data does not exist
                objectData = [float(x) for x in data]
                print('Object name: ', objNames[i], "Object Data: " ,objectData)
                objects.append(objectData)
            # writeData (objectData, objeNames[i])
        else:
            continue

class csvConverterVICON:

    def __init__(self, fileName):
        fileName.self = fileName
        print("Init Vicon file", fileName)
        if os._exists(fileName):
            self.generateNewFileName(fileName)
            self.loadFile(fileName)
        else:
            print(" The file does not exist ")

    def generateNewFileName(self, fileName):
        print("Current File Name", fileName)
        dirName = os.path.dirname(fileName)
        baseName = os.path.basename(fileName)
        fileNameWoExt, extension = os.path.splitext(baseName)
        newFileName = os.path.join(dirName, fileNameWoExt + '_converted' + extension )
        print(newFileName)

    def loadFile(self,fileName):
        print("Loading file for reading")

    def writeFile(self):
        """File writing function"""
        print("Write in file")

def main():
    dirName = "D:\BirdTrackingProject\VICON_DataRead"
    fileName = "ErwinCarriesObject_02.csv"
    csvFile = os.path.join(dirName, fileName)
    # csvFile = os.path.join(dirName, "testData.csv")
    print("Name of csv file : ", csvFile)

    fileConverter = csvConverterVICON()

    with open(csvFile, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 2:
                print("No of Objects : ", math.floor((len(row) - 2) / 6))
                objNames = getObjectNames(row)
            if line_count > 4:
                getRowData(row, objNames)
            line_count += 1
    print("Main Function")

if __name__ == '__main__':
    main()