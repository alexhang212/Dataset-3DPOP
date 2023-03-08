#import xml.etree.ElementTree as ET
from lxml import etree as ET
import os


class xmlSettingsParser:

    def __init__(self, filePath, ObjectID=None):
        """
        Initialze the settings xml class
        :param rootDir: directory having the settings file
        :param fileName: name of the settings file
        :param ObjectID: List of names of objects defined in the trial (e.g bird IDs)
        """

        self.filePath = filePath
        # If the file does not exist we create a mock file
        if not os.path.exists(self.filePath):
            self.rootDir = os.path.dirname(self.filePath)
            self.fileName = os.path.basename(self.filePath)
            self.parentDir = os.path.basename(self.rootDir)
            self.mode = "Vicon" #mode for setting, "custom for the use of custom cameras, Vicon for Vicon VUEs"
            print("File does not exist -- creating Mock File @: ", self.filePath)
            self.createMockFile(ObjectID)

        # Empty dict for settings file
        self.settingsDict = {}
        # If the file exists we load it
        if os.path.exists(self.filePath):
            self.readFile()

    def createMockFile(self,SubjectID):
        """
        Create a mock settings file for the given session
        :return: None
        """
        print("File Name : ", self.filePath)
        # Extract Name of root directory
        fileNamewoExt = os.path.splitext(self.fileName)[0]
        self.sessionName = fileNamewoExt.split("settings_")[1]

        root = ET.Element("settings")

        # Add project related information
        files = ET.SubElement(root, "projectInfo")
        ET.SubElement(files, "rootDirectory").text = self.rootDir
        ET.SubElement(files, "session").text = self.sessionName
        ET.SubElement(files,"Mode").text = self.mode

        # Add camera related info
        doc = ET.SubElement(root, "cameras")
        cameraIDs = ["2118670","2119571","2122204","2122725"]
        for camid in cameraIDs:
            ET.SubElement(doc, "cameraID").text = camid

        object = ET.SubElement(root, "objectsToTrack")
        if SubjectID is not None:
            for subject in SubjectID:
                for type in ["bp","hd"]:
                    ET.SubElement(object,"trackingObject").text = "%s_%s"%(subject,type)
        else:
            ET.SubElement(object, "trackingObject").text = "bp" #Backpack and head
            ET.SubElement(object, "trackingObject").text = "hd"

        subjects = ET.SubElement(root, "Subjects")
        if SubjectID is not None:
            for object in SubjectID:
                ET.SubElement(subjects,"SubjectID").text = object

        files = ET.SubElement(root, "files")
        dataFile = str(self.sessionName)
        ET.SubElement(files, "dataFile").text =  dataFile + ".csv"
        ET.SubElement(files, "dataFile3D").text = dataFile + ".3D" + ".csv"
        ET.SubElement(files, "calibFile").text = dataFile + ".xcp"
        ET.SubElement(files, "c3dFileStream").text = dataFile + ".c3d"
        ET.SubElement(files, "framesToCaptureFile").text =  dataFile + ".framesToCapture.txt"
        ET.SubElement(files, "customFeatureFile").text = "customFeatures.txt"
        ET.SubElement(files, "customFeatureFile3D").text = dataFile + ".customFeatures" + ".txt"
        ET.SubElement(files, "MatlabCSVFile").text = dataFile + "_Matlab.csv"
        ET.SubElement(files, "FinalFeatureCSV").text = dataFile + "_FinalFeature.csv"
        ET.SubElement(files,"FinalFeatureCSV2D").text = dataFile + "_FinalFeature2D"
        ET.SubElement(files,"FinalFeatureCSVBBox").text = dataFile + "_FinalFeatureBBox"

        files = ET.SubElement(root, "videoFiles")
        for camid in cameraIDs:
            fileName = str(self.sessionName) + "." + camid
            ET.SubElement(files, "videoFile").text = os.path.join( fileName + ".avi")

        files = ET.SubElement(root, "annotationFiles")
        for camid in cameraIDs:
            fileName = str(self.sessionName) + "." + camid
            ET.SubElement(files, "annotationFile").text = os.path.join(fileName + ".csv")

        files = ET.SubElement(root, "annotationDataBaseFiles")
        for camid in cameraIDs:
            fileName = str(self.sessionName) + "." + camid + ".database"
            ET.SubElement(files, "dataBaseFile").text = os.path.join(fileName + ".csv")

        files = ET.SubElement(root, "videoReadoutSettings")
        ET.SubElement(files, "startFrame").text = "0"
        ET.SubElement(files, "endFrame").text = "0"
        ET.SubElement(files, "stepSize").text = "100"


        tree = ET.ElementTree(root)
        tree.write(self.filePath, pretty_print="True")

    def readFile(self):

        tree = ET.parse(self.filePath)
        root = tree.getroot()

        for elem in root:
            print(elem.tag)
            if elem.tag == "cameras":
                camera  = []
                for subelem in elem:
                    # print(" camera : ", subelem.text)
                    camera.append(subelem.text)
                self.settingsDict[elem.tag] = camera
            elif elem.tag == "objectsToTrack" :
                objects = []
                for subelem in elem:
                    # print("videoReadoutSettings: ", subelem.text)
                    objects.append(subelem.text)
                self.settingsDict[elem.tag] = objects
            elif elem.tag == "Subjects" :
                objects = []
                for subelem in elem:
                    # print("videoReadoutSettings: ", subelem.text)
                    objects.append(subelem.text)
                self.settingsDict[elem.tag] = objects
            elif elem.tag == "annotationFiles":
                objects = []
                for subelem in elem:
                    print("annotationFiles: ", subelem.text)
                    objects.append(subelem.text)
                self.settingsDict[elem.tag] = objects

            elif elem.tag == "annotationDataBaseFiles":
                objects = []
                for subelem in elem:
                    print("annotationDataBaseFiles: ", subelem.text)
                    objects.append(subelem.text)
                self.settingsDict[elem.tag] = objects

            elif elem.tag == "videoFiles":
                objects = []
                for subelem in elem:
                    # print("videoFiles: ", subelem.text)
                    objects.append(subelem.text)
                self.settingsDict[elem.tag] = objects
            elif elem.tag == "files":
                for subelem in elem:
                    # print(" file names : ", subelem.text)
                    self.settingsDict[subelem.tag] = subelem.text
            elif elem.tag == "projectInfo" :
                for subelem in elem:
                    # print("project info: ", subelem.text)
                    self.settingsDict[subelem.tag] = subelem.text
            elif elem.tag == "videoReadoutSettings" :
                for subelem in elem:
                    # print("videoReadoutSettings: ", subelem.text)
                    self.settingsDict[subelem.tag] = subelem.text
            else :
                print("No matching tag")

        def getFullPath(fileName , rootDir):
            """
            Get full path for the given file based on the given root dir
            :param fileName: str
            :param rootDir: str
            :return: Combine path to give full path
            """
            return os.path.join(rootDir, fileName)

        def updateNode(nodeString, settingsDict):
            """
            Update the given node based on the given settings dict
            :param nodeString: str
            :param settingsDict: dict
            :return: None
            """
            tree = ET.parse(self.filePath)
            root = tree.getroot()

            for elem in root:
                print(elem.tag)
                if elem.tag == "cameras":
                    for subelem in elem:
                        # print(" camera : ", subelem.text)
                        camera.append(subelem.text)
                    self.settingsDict[elem.tag] = camera

        def addNode(nodeString, settingsDict):
            """
            Add node based on the values given from the settings dict
            :param nodeString: str
            :param settingsDict: dict
            :return: None
            """
            print("Adding new node")



def main():

    filePath = "D:\\BirdTrackingProject\\20190618_PigeonPostureDataset\\settings_session02_test.xml"
    print("Loading Default File Path : ", filePath)
    xmlParser = xmlSettingsParser(filePath)
    settingsDict = xmlParser.__getattribute__("settingsDict")
    print(settingsDict)

if __name__ == '__main__':
    main()



