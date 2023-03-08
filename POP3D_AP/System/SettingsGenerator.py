""" Settings class for 3D POP AP for 3DPOP dataset"""



#import xml.etree.ElementTree as ET
from lxml import etree as ET
import os


class xmlSettingsParser:

    def __init__(self, SettingsPath,InputTrial):
                #  filePath,CustomVideoName=None, ObjectID=None):
        """
        Initialze the settings xml class
        :param SettingsPath: path to settings file
        :param InputTrial: Trial Object
        """

        self.filePath = SettingsPath
        self.Trial = InputTrial
        self.TrialDir = self.Trial.baseDir
        
        # If the file does not exist we create a mock file
        if not os.path.exists(self.filePath):
            self.mode = "Custom"
            print("File does not exist -- creating Mock File @: ", self.filePath)
            self.createMockFile()

        # Empty dict for settings file
        self.settingsDict = {}
        # If the file exists we load it
        if os.path.exists(self.filePath):
            self.readFile()

    def createMockFile(self):
        """
        Create a mock settings file for the given session
        :return: None
        """
        print("File Name : ", self.filePath)
        # Extract Name of root directory
        self.TrialName = self.Trial.TrialName
        # import ipdb;ipdb.set_trace()
        
        # fileNamewoExt = os.path.splitext(self.fileName)[0]
        # self.sessionName = fileNamewoExt.split("settings_")[1]

        root = ET.Element("settings")

        # Add project related information
        files = ET.SubElement(root, "projectInfo")
        ET.SubElement(files,"Mode").text = self.mode
        ET.SubElement(files, "rootDirectory").text = self.TrialDir
        ET.SubElement(files, "session").text = self.TrialName
        ET.SubElement(files, "ResolutionHeight").text = "2160"
        ET.SubElement(files,"ResolutionWidth").text = "3840"
        ET.SubElement(files,"AnnotationDirectory").text = os.path.join(self.TrialDir,"POP3DAP_Data","Annotation")
        ET.SubElement(files,"DataDirectory").text = os.path.join(self.TrialDir,"POP3DAP_Data","Data")


        # Add camera related info
        doc = ET.SubElement(root, "cameras")
        cameraIDs = ["Cam1","Cam2","Cam3","Cam4"]
        for camid in cameraIDs:
            ET.SubElement(doc, "cameraID").text = camid

        object = ET.SubElement(root, "objectsToTrack")
        for subject in self.Trial.Subjects:
            for type in ["bp","hd"]:
                ET.SubElement(object,"trackingObject").text = "%s_%s"%(subject,type)

        subjects = ET.SubElement(root, "Subjects")
        for object in  self.Trial.Subjects:
            ET.SubElement(subjects,"SubjectID").text = object

        files = ET.SubElement(root, "files")
        dataFile = str(self.TrialName)
        ET.SubElement(files, "dataFile").text =  dataFile + ".csv"
        ET.SubElement(files, "dataFile3D").text = dataFile + ".3D" + ".csv"
        ET.SubElement(files, "framesToCaptureFile").text =  dataFile + ".framesToCapture.txt"
        ET.SubElement(files, "customFeatureFile").text = "customFeatures.txt"
        ET.SubElement(files, "customFeatureFile3D").text = dataFile + ".customFeatures" + ".txt"
        ET.SubElement(files, "MatlabCSVFile").text = dataFile + "_Matlab.csv"

        files = ET.SubElement(root, "videoFiles")
        for x in range(len(cameraIDs)):
            ET.SubElement(files, "videoFile").text = self.Trial.camObjects[x].VideoPath

        files = ET.SubElement(root, "TrialName")
        for camid in cameraIDs:
            fileName = camid + "_" + str(self.TrialName)  
            ET.SubElement(files, "TrialName").text = os.path.join(fileName + ".csv")
            
        files = ET.SubElement(root, "annotationFiles")
        for camid in cameraIDs:
            fileName = camid + "_" + str(self.TrialName)  
            ET.SubElement(files, "annotationFile").text = os.path.join(fileName + ".csv")

        files = ET.SubElement(root, "annotationDataBaseFiles")
        for camid in cameraIDs:
            fileName = camid + "_" + str(self.TrialName) + ".database"
            ET.SubElement(files, "dataBaseFile").text = os.path.join(fileName + ".csv")
        ###
        files = ET.SubElement(root, "FinalFeatureCSV")
        fileName = dataFile + "_FinalFeature"
        ET.SubElement(files, "FinalFeatureCSV").text = os.path.join(fileName + ".csv")
            
        files = ET.SubElement(root, "FinalFeatureCSV3D")
        for camid in cameraIDs:
            fileName = dataFile + "-" + camid + "-Keypoint3D"
            ET.SubElement(files, "FinalFeatureCSV3D").text = os.path.join(fileName + ".csv")

        files = ET.SubElement(root, "FinalFeatureCSV2D")
        for camid in cameraIDs:
            fileName = dataFile + "-" + camid + "-Keypoint2D"
            ET.SubElement(files, "FinalFeatureCSV2D").text = os.path.join(fileName + ".csv")
            
        files = ET.SubElement(root, "FinalFeatureCSVBBox")
        for camid in cameraIDs:
            fileName = dataFile + "-" + camid + "-BBox"
            ET.SubElement(files, "FinalFeatureCSVBBox").text = os.path.join(fileName + ".csv")

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

            elif elem.tag == "FinalFeatureCSV" :
                for subelem in elem:
                    # print("videoReadoutSettings: ", subelem.text)
                    self.settingsDict[subelem.tag] = subelem.text
                                        
            elif elem.tag == "FinalFeatureCSV3D" :
                objects = []
                for subelem in elem:
                    objects.append(subelem.text)
                    # print("videoReadoutSettings: ", subelem.text)
                    self.settingsDict[subelem.tag] = objects
            elif elem.tag == "FinalFeatureCSV2D" :
                objects = []
                for subelem in elem:
                    objects.append(subelem.text)
                    # print("videoReadoutSettings: ", subelem.text)
                    self.settingsDict[subelem.tag] = objects
            elif elem.tag == "FinalFeatureCSVBBox" :
                objects = []
                for subelem in elem:
                    objects.append(subelem.text)
                    # print("videoReadoutSettings: ", subelem.text)
                    self.settingsDict[subelem.tag] = objects
            else :
                print("No matching tag")

def main():

    filePath = "D:\\BirdTrackingProject\\20190618_PigeonPostureDataset\\settings_session02_test.xml"
    print("Loading Default File Path : ", filePath)
    xmlParser = xmlSettingsParser(filePath)
    settingsDict = xmlParser.__getattribute__("settingsDict")
    print(settingsDict)

if __name__ == '__main__':
    main()



