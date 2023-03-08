import cv2 as cv
import os
from System import SettingsGenerator
import numpy as np


def setWindowName(path):
    windowName = os.path.basename(path)
    windowName = windowName.split(".avi")[0]
    return windowName

class VideoVicon:
    # todo : Global variable must have serialNo savedfor each object, so we can check declaration of same camera
    def __init__(self, path, serialNo):
        """
        Initialization of class, gets path for the video files and return images
        """
        if os.path.exists(path):
            self.videoFilePath = path
            self.capture = cv.VideoCapture(self.videoFilePath)
            self.totalFrameCount = int(self.capture.get(cv.CAP_PROP_FRAME_COUNT))
            self.windowName = setWindowName(self.videoFilePath)
            self.objectID = serialNo
        else:
            raise ValueError("Image path does not exist")


    def getFrame(self, frameNo):
        """
        returns the image for given frame number in the video
        :param frameNo: Required frame no
        :return: image
        """
       # capture = cv.VideoCapture(self.videoFilePath)
        # Sanity check for the given frame number
        frameNo = np.floor(frameNo)
        if frameNo <= self.totalFrameCount:
            self.capture.set(cv.CAP_PROP_POS_FRAMES, frameNo)
        else:
            raise ValueError(" Frame no does not exist")

        ret, frame = self.capture.read()

        return frame

    def __del__(self):
        del self.capture

def makeEntryToFile(fileName, frameNo):
    assert (os.path.exists(fileName)), "File does no exist"
    file = open(fileName, 'a')
    file.write(str(frameNo))
    file.write("\n")
    file.close()


def defineVideoFramesForAnnotation(settingsDict):
    """
    Author: Alex Chan
    Reads in N camera views

    Reads the settings dict and prepares the video files for annotation, i.e. allows the user to select good frames for annotation of features.
    The output is a file generated with name framesToCaptureFile.txt which will be followed by the annotation program to select frames for annotation.
    :param settingsDict: Dictionary
    :return: None
    """
    videoFiles = settingsDict["videoFiles"]
    DataDir = settingsDict["DataDirectory"]
    rootDir = settingsDict["rootDirectory"]
    # Iterate through path anf provide global path
    for file in range(len(videoFiles)):
        videoFiles[file] = os.path.join(rootDir, videoFiles[file])

    fileName = settingsDict["framesToCaptureFile"]
    fileName = os.path.join(DataDir, fileName)

    windowNames = []
    videoObjects = []
    stepSize = 100

    if os.path.exists(fileName):
        file = open(fileName, 'a')
        file.close()
    else:
        # If the file does not exist then it is created
        file = open(fileName, 'w')
        file.close()


    for i in range(len(videoFiles)):
        videoObjects.append(VideoVicon(videoFiles[i], i))

    frameCounts = [None]*len(videoObjects)

    for i in range(len(videoObjects)):
        frameCounts[i] = videoObjects[i].totalFrameCount


    cv.namedWindow("temp", cv.WINDOW_NORMAL)


    #Check if all videos have same number of frames
    # assert (len(set(frameCounts))==1), " Video files do not have same frame number"
    frameNo = 0

    # captureStatus= False

    while 0<= frameNo <= frameCounts[0]:
        images = []
        print("Frame No: ", frameNo)

        for j in range(len(videoObjects)):
            image = videoObjects[j].getFrame(frameNo)

            images.append(image)

        #determines how the images are split, 2 images per row
        SplitIndex = [[k, k+1] for k in range(0,len(images),2)]

        combinedImage = np.concatenate([np.concatenate([images[x[0]],images[x[1]]],axis=1 ) for x in SplitIndex], axis=0)

        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(combinedImage, "frameCount : " + str(frameNo) + "/" + str(frameCounts[0]), (10, 50), font, 1,
                   (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(combinedImage, "Step Size : " + str(stepSize), (10, 150), font, 1,
                   (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(combinedImage, "Press 'h' for help", (10, 250), font, 1,
                   (255, 255, 255), 2, cv.LINE_AA)

        cv.imshow("temp", combinedImage)

        k = cv.waitKey(10)
        if k == ord('+'):
            stepSize += 100

        if k == ord('-'):
            stepSize -= 50

        if k == ord('q'):
            cv.destroyAllWindows()
            break
        if k == ord('n'):
            frameNo += stepSize
            continue
        if k == ord('b'):
            frameNo -= stepSize
        if k == ord('s'):
            makeEntryToFile(fileName, frameNo)  # ,captureStatus)

        if k == ord('h'):
            print("+ : increase step size by 100 \n")
            print("- : increase step size by 50 \n ")
            print("n : Next frame \n ")
            print("b : previous frame \n ")
            print("S : Enter frame status in log file \n ")

    print("Program terminated frame No/totalFrame : ", frameNo, "/", frameCounts[0])





def main(DatasetDir, SequenceNum):
    # Example will use this class to make small application of viewing the video files
    settingsObject = settingsGenerator.xmlSettingsParser(DatasetDir, SequenceNum)
    defineVideoFramesForAnnotation(settingsObject.settingsDict)

if __name__ == "__main__":
    #run default
    DatasetDir = "/media/alexchan/My Passport/Pop3D-Dataset_Final/"
    SequenceNum = 1
    main(DatasetDir, SequenceNum)
