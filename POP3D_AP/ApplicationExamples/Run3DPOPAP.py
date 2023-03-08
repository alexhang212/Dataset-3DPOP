""""
Author: Alex Chan
Annotating multiple pigeons within 1 trial

The file is created to start your own annotation sequence for computing 3D position of custom features.
The code follows following steps.
1. First selection of frames for annotation (Implemented in AnnotationFrameSelector.Py)
2. Opens annotating tool to capture 2D positions for each subject
3. 2D positions from different images are triangulated and 3D positions w.r.t corresponding 6-DOF pattern  are saved in separate file.
4. Propagates the 3D position of custom features to the whole trial and output the 3D coordinates of the features
5. Generates 2D and bounding box ground truth from 3D data.
"""

import sys
sys.path.append('../')
sys.path.append('./')


from System import videoVicon
from DrawingOperations import imageAnnotation
from ApplicationExamples import createFeaturesUsingAnnotation, ComputePlanes
from System import SettingsGenerator
from GenerateAnnotation import Generate3DKeypoints, Generate2DKeypoints, GenerateBBox,GenerateSubject2DKeypoints
import os
from POP3D_Reader import Trial




def main(DatasetDir,SequenceNum):
    """Main function to run annotation pipeline"""
    
    ## Read Trial
    InputTrial = Trial.Trial(DatasetDir,SequenceNum)
    
    ##Generate/Read Settings file
    SettingsPath = os.path.join(InputTrial.baseDir, "POP3DAP_Data","%s_settings.xml"%InputTrial.TrialName)
    projectSettings = SettingsGenerator.xmlSettingsParser(SettingsPath,InputTrial)
    settingsDict = projectSettings.settingsDict
    
    # # # # # Select the frames for annotation ( without projection of the markers on the image) --
    videoVicon.defineVideoFramesForAnnotation(settingsDict)

    # # # # #Invoke the annotation tool, allows annotation of features based on the custom features file defined in the settings file.
    annotationToolObject = imageAnnotation.imageAnnotationTool_Multi(projectSettings)
    annotationToolObject.run()

    # # Triangulate the feature created using the provided annotation and show projection on the image to accept or reject the
    # # annotation. The back projection must align with the features in both the images.
    createFeaturesUsingAnnotation.createTriangulatedFeatures(projectSettings)

    # #get the obtained features then applies is accross the whole trial to get 3D coordinates of keypoints, save as csv
    Generate3DKeypoints.Generate3DKeypoint(projectSettings)
    Generate2DKeypoints.Generate2DKeypoint(projectSettings)
    GenerateBBox.GenerateBBox(projectSettings)
    # GenerateSubject2DKeypoints.GenerateSubject2DKeypoints(projectSettings,imsize=(320,320))
    

if __name__ == "__main__":
    
    ##Default:
    DatasetDir = "/media/alexchan/My Passport/Pop3D-Dataset_Final/" #put path to dataset here
    SequenceNum = 1

    main(DatasetDir, SequenceNum)

