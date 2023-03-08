""""
Author: Alex Chan

The file is created to start your own annotation sequence for computing 3D position of custom features.
The code follows following 3 steps.
1. First selection of frames for annotation (Implemented in AnnotationFrameSelector.Py)
2. Opens annotating tool to capture 2D positions for each subject
3. 2D positions from different images are triangulated and 3D positions w.r.t corresponding 6-DOF pattern  are saved in separate file.
4. (NEW) Applies the 3D position of custom features to the whole trial and output the 3D coordinates of the features
"""

import sys
sys.path.append('./')

from System import videoVicon
from DrawingOperations import imageAnnotation
from VICONApplicationExamples import createFeaturesUsingAnnotation
from FileOperations import settingsGenerator
from VICONGenerateTraining import Generate3DKeypoints, Generate2DKeypoints, GenerateBBox,GenerateSubject2DKeypoints



# projectSettings = settingsGenerator.xmlSettingsParser("/home/alexchan/Documents/MAAP3D/Trials/01072022Pigeon10/Vicon/settings_training_01072022_02.xml")
# settingFile = "/home/alexchan/Documents/MAAP3D/Trials/01072022Pigeon10/Vicon/settings_training_01072022_02.xml"
# settingFile = "/home/alexchan/Documents/MAAP3D/Trials/01072022Pigeon10/Vicon/settings_training_01072022_03.xml"

# projectSettings = settingsGenerator.xmlSettingsParser(settingFile,["452_0107"])


def main(settingFile,TrackedSubjects):

    # # Settings generator to create settings file if not available or read settings file
    settingFile = "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/29062022Pigeon10/Vicon/settings_training_29062022_07.xml"
    # TrackedSubjects = ["475_2906","382_2906"]
    TrackedSubjects =[1]

    projectSettings = settingsGenerator.xmlSettingsParser(settingFile,TrackedSubjects)
    settings = projectSettings.settingsDict

    # # # # Select the frames for annotation ( without projection of the markers on the image) --
    # # # # todo : Add the implementation from annotationFrameSelector
    # videoVicon.defineVideoFramesForAnnotation_NCam(settings)

    # # # #Invoke the annotation tool, allows annotation of features based on the custom features file defined in the settings file.
    # annotationToolObject = imageAnnotation.imageAnnotationTool_Multi(projectSettings)
    # annotationToolObject.run()

    # Triangulate the feature created using the provided annotation and show projection on the image to accept or reject the
    # annotation. The back projection must align with the features in both the images.
    # createFeaturesUsingAnnotation.createTriangulatedFeatures(projectSettings)

    # Generate3DKeypoints.Generate3DKeypoint(projectSettings)


    ###Extract 2D, BBox and subject wise video
    # Generate2DKeypoints.Generate2DKeypoint(projectSettings)
    # GenerateBBox.GenerateBBox(projectSettings)
    GenerateSubject2DKeypoints.GenerateSubject2DKeypoints(projectSettings,imsize = (160,160))

    

if __name__ == "__main__":
    # defaultSettingFile = "/home/alexchan/Documents/MAAP3D/Trials/01072022Pigeon10/Vicon/settings_training_01072022_02.xml"
    main("yo","wow")

    ###Rerun high memory stuff
    # settingFileList =["/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/28062022Pigeon10/Vicon/settings_training_28062022_06.xml"]
    # TrackedSubjectsList = [1]


    # for i in range(1):
    #     main(settingFileList[i],TrackedSubjectsList[i])



    ####01072022#####
    # settingFileList =["/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/01072022Pigeon10/Vicon/settings_training_01072022_02.xml",
    # "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/01072022Pigeon10/Vicon/settings_training_01072022_03.xml",
    # "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/01072022Pigeon10/Vicon/settings_training_01072022_04.xml",
    #     "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/01072022Pigeon10/Vicon/settings_training_01072022_05.xml",
    #                     "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/01072022Pigeon10/Vicon/settings_training_01072022_06.xml",
    #                             "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/01072022Pigeon10/Vicon/settings_training_01072022_09.xml",
    #                                 "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/01072022Pigeon10/Vicon/settings_training_01072022_10.xml",
    #                                "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/01072022Pigeon10/Vicon/settings_training_01072022_11.xml",
    #                              "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/01072022Pigeon10/Vicon/settings_training_01072022_12.xml",
    #                              "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/01072022Pigeon10/Vicon/settings_training_01072022_13.xml",
    #                            "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/01072022Pigeon10/Vicon/settings_training_01072022_14.xml"
    # ]

    # TrackedSubjectsList = [ ["452_0107"],
    # ["452_0107","484_0107"],
    # ["391_0107"],
    # ["391_0107","473_0107"],
    # ["391_0107","473_0107","54_0107","452_0107","484_0107"],
    # ["705_0107"],
    # ["705_0107","486_0107"],
    # ["485_0107"],
    # ["485_0107","47_0107"],
    # ["485_0107","47_0107","486_0107","487_0107","705_0107"],
    # ["47_0107","54_0107","391_0107","452_0107","473_0107","484_0107","485_0107","486_0107","487_0107","705_0107"]
    # ]


    # for i in range(11):
    #     main(settingFileList[i],TrackedSubjectsList[i])









    ##13072022##
    # settingFileList =["/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/13072022Pigeon10/Vicon/settings_training_13072022_03.xml",
    # "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/13072022Pigeon10/Vicon/settings_training_13072022_04.xml",
    # "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/13072022Pigeon10/Vicon/settings_training_13072022_05.xml",
    #     "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/13072022Pigeon10/Vicon/settings_training_13072022_06.xml",
    #                     "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/13072022Pigeon10/Vicon/settings_training_13072022_11.xml",
    #                         "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/13072022Pigeon10/Vicon/settings_training_13072022_12.xml",
    #                             "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/13072022Pigeon10/Vicon/settings_training_13072022_13.xml",
    #                                 "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/13072022Pigeon10/Vicon/settings_training_13072022_14.xml"
    # ]

    # TrackedSubjectsList = [["497_1307"],
    #                         ["497_1307","483_1307"],
    #                         ["707_1307"],
    #                         ["707_1307","485_1307"],
    #                         ["706_1307"],
    #                         ["706_1307","382_1307"],
    #                         ["706_1307","382_1307","473_1307","475_1307","391_1307"],
    #                         ["382_1307","391_1307","473_1307","475_1307","483_1307","485_1307","486_1307","497_1307","706_1307","707_1307"]
    #                         ]

    # for i in range(8):
    #     main(settingFileList[i],TrackedSubjectsList[i])



    # ### 04072022##
    # settingFileList =["/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/04072022Pigeon10/Vicon/settings_training_04072022_02.xml",
    # "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/04072022Pigeon10/Vicon/settings_training_04072022_04.xml",
    # "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/04072022Pigeon10/Vicon/settings_training_04072022_05.xml",
    # "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/04072022Pigeon10/Vicon/settings_training_04072022_06.xml",
    # "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/04072022Pigeon10/Vicon/settings_training_04072022_07.xml",
    # "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/04072022Pigeon10/Vicon/settings_training_04072022_09.xml",
    # "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/04072022Pigeon10/Vicon/settings_training_04072022_10.xml",
    # "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/04072022Pigeon10/Vicon/settings_training_04072022_11.xml",
    # "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/04072022Pigeon10/Vicon/settings_training_04072022_12.xml",
    # "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/04072022Pigeon10/Vicon/settings_training_04072022_13.xml",
    # "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/04072022Pigeon10/Vicon/settings_training_04072022_14.xml"
    # ]
    


    # TrackedSubjectsList = [["389_0407"],
    #                         ["389_0407","47_0407"],
    #                         ["391_0407"],
    #                         ["391_0407","497_0407"],
    #                         ["391_0407","497_0407","47_0407","389_0407","705_0407"],
    #                         ["452_0407"],
    #                         ["452_0407","382_0407"],
    #                         ["706_0407"],
    #                         ["706_0407","707_0407"],
    #                         ["706_0407","707_0407","475_0407","452_0407","382_0407"],
    #                         ["47_0407","382_0407","389_0407","391_0407","452_0407","475_0407","497_0407","705_0407","706_0407","707_0407"]
    #                         ]
    
    # for i in range(11):
    #     main(settingFileList[i],TrackedSubjectsList[i])



    ##30062022
    # settingFileList =["/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/30062022Pigeon10/Vicon/settings_training_30062022_02.xml",
    # "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/30062022Pigeon10/Vicon/settings_training_30062022_03.xml",
    # "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/30062022Pigeon10/Vicon/settings_training_30062022_05.xml"
    # ]

    # TrackedSubjectsList = [ ["483_3006","706_3006"],
    # ["452_3006","486_3006","487_3006","483_3006","706_3006"],
    # ["391_3006","54_3006"] ]

    # for i in range(3):
    #     main(settingFileList[i],TrackedSubjectsList[i])

    ###29062022:
    # settingFileList =["/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/29062022Pigeon10/Vicon/settings_training_29062022_03.xml",
    # "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/29062022Pigeon10/Vicon/settings_training_29062022_05.xml",
    # "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/29062022Pigeon10/Vicon/settings_training_29062022_07.xml",
    # "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/29062022Pigeon10/Vicon/settings_training_29062022_08.xml"
    # ]

    # TrackedSubjectsList = [ ["475_2906","382_2906"],
    # ["483_2906","705_2906","473_2906","382_2906","475_2906"],
    # ["484_2906","708_2906"],
    # ["47_2906","382_2906","389_2906","473_2906","475_2906","483_2906","484_2906","705_2906","707_2906","708_2906"]
    # ]

    # for i in range(4):
    #     main(settingFileList[i],TrackedSubjectsList[i])

    # main(settingFileList[3],TrackedSubjectsList[3])

    # ###28062022
    # settingFileList =["/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/28062022Pigeon10/Vicon/settings_training_28062022_02.xml",
    # "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/28062022Pigeon10/Vicon/settings_training_28062022_03.xml",
    # "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/28062022Pigeon10/Vicon/settings_training_28062022_05.xml",
    # "/run/user/1000/gvfs/smb-share:server=kanonas.local,share=homes/alexhang/Backups/28062022Pigeon10/Vicon/settings_training_28062022_06.xml"
    # ]

    # TrackedSubjectsList = [ ["473_2806","708_2806"],
    # ["706_2806","389_2806","486_2806","708_2806","473_2806"],
    # ["47_2806","707_2806"],
    # ["47_2806","475_2806","487_2806","389_2806","473_2806","483_2806","486_2806","706_2806","708_2806","707_2806"]
    # ]

    # for i in range(4):
    #     main(settingFileList[i],TrackedSubjectsList[i])