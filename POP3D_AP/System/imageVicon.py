from cmath import inf
from Math import imageOperations as imageOp
from Math import transformations as transformationOp
import numpy as np
import cv2 as cv
from DrawingOperations import drawOp
import os


class ImageVicon:
    """
    Class handles images collected from vicon videos, computations of image projections and saving 2D features
    """
    def __init__(self,
                 objectID,
                 distortionMatrix = np.zeros((1,5)),
                 instrinsicMatrix = np.identity(3),
                 customObjectsDim = None
                 ):

        self.objectId = objectID # To stores camera
        self.imageHeight = 1080 # rows
        self.imageWidth = 1920 # cols

        if customObjectsDim != None:
            self.imageHeight = customObjectsDim[1] # rows
            self.imageWidth = customObjectsDim[0] # cols

        assert(distortionMatrix.shape == (1,5)),"Shape mismatch : Distortion matrix"
        self.distortionMatrix = distortionMatrix

        assert (instrinsicMatrix.shape == (3, 3)),"Shape mismatch : Instrinsic matrix"
        self.intrinsicMatrix = instrinsicMatrix

        self.featureDict = {}

    def drawFeatureLine(self,image,point1,point2,lineColor = (255,255,255), lineWidth = 3):
        """
        Draw line between given features
        :return:
        """
        cv.line(image, (int(point1[0]), int(point1[1])),
                (int(point2[0]), int(point2[1])), lineColor , lineWidth)


    def drawPosture(self, image):
        # HYR =(Height, Y coordinate, Rows), WXC = (Width, X coordinate, Cols)

        # Draw head Vectors
        if "head_beak" in self.featureDict and "head_leftEye" in self.featureDict:
            self.drawFeatureLine(image, self.featureDict["head_beak"],self.featureDict["head_leftEye"],(0,0,255))
        if "head_beak" in self.featureDict and "head_rightEye" in self.featureDict:
            self.drawFeatureLine(image, self.featureDict["head_beak"],self.featureDict["head_rightEye"],(0,255,255))

        if "body_tail" in self.featureDict and "body_leftShoulder" in self.featureDict:
            self.drawFeatureLine(image, self.featureDict["body_tail"],self.featureDict["body_leftShoulder"],(255,255,0))
        if "body_tail" in self.featureDict and "body_rightShoulder" in self.featureDict:
            self.drawFeatureLine(image, self.featureDict["body_tail"],self.featureDict["body_rightShoulder"],(255,0,255))


    def getRoiOnKeypoint(self,feature, tempImage, roiSize = 400):

        point = [0,0]
        clone = tempImage.copy()
        height, width, channel = clone.shape

        if feature in self.featureDict:
            point = self.featureDict[feature]
            point = [int(point[0]),int(point[1])]

        resized = np.zeros((400,400,3), np.uint8)
        if (point[0] != 0 and point[1] != 0) and imageOp.isPointValid(point, height,width):
            roi = self.getRoi(point, height, width, 100)
            roiWidth = roi[3] - roi[2]  # x2 - x1
            roiHeight = roi[1] - roi[0]  # y2 - y1
            roiImage = clone[roi[2]:roi[3], roi[0]:roi[1], :]
            resized = cv.resize(roiImage, (400, 400), interpolation=cv.INTER_AREA)

        # Fill in black frame if no feature was detected
        #clone[0:resized.shape[0], 0:resized.shape[1]] = resized

        return resized

    def getRoi(self, point , height, width , size = 200):
        x = point[0]
        y = point[1]
        roi = [0 , 0, 200, 200]

        roi[0] = x - size
        if roi[0] < 0:
            roi[0] = 0

        roi[1] = x + size
        if roi[1] > width:
            roi[1] = width - 1

        roi[2] = y - size
        if roi[2]< 0:
            roi[2] = 0

        roi[3]= y + size
        if roi[3] > height:
            roi[3] = height -1

        return roi


    def computeSubjectBoundingBox(self, subject,offset = 40):
        """
        Author: Alex Chan
        Compute the corners required for drawing the bounding box for single subject\
        Looks for keypoint with the most extreme values
        Note: output is little different from parent function, corners output is always
        top left corner and bottom right corner
        :return:
        """
        # import ipdb; ipdb.set_trace()

        bBox = {}
        ###compute leftmost, rightmost, topmost and bottommost point
        leftmost = float("Inf")
        rightmost = -float("Inf")
        topmost = float("Inf")
        bottommost = -float("Inf")

        SubjectDict = {k:v for k,v in self.featureDict.items() if k.startswith(subject)}

        for point in SubjectDict.values():
            if point == [0,0]:
                continue
            else:#Update points if the point is larger/smaller than most extreme point so far
                if point[0] < leftmost:
                    leftmost = point[0]
                if point[0] > rightmost:
                    rightmost = point[0]
                if point[1] < topmost:
                    topmost= point[1]
                if point[1] > bottommost:
                    bottommost = point[1]

        # import ipdb; ipdb.set_trace()

        lCorner = [leftmost - offset, topmost - offset]
        rCorner = [rightmost + offset, bottommost + offset]
        lCorner = [0 if point <0 else point for point in lCorner]
        rCorner = [0 if point <0 else point for point in rCorner]


        bBox["lCorner"] = lCorner
        bBox["rCorner"] = rCorner

        return bBox

    def computeBoundingBox(self):
        """
        Compute the corners required for drawing the bounding box
        :return:
        """
        bBox = {}
        if "head_beak" in self.featureDict:
            head = self.featureDict["head_beak"]
        else:
            head = [0, 0]

        if "body_tail" in self.featureDict:
            tail = self.featureDict["body_tail"]
        else:
            tail = [0, 0]

        offset = 40
        # if head is left of the tail
        if head != [0, 0] and tail != [0, 0]:
            if head[0] < tail[0]:
                lCorner = [head[0] - offset, head[1] - offset]
                rCorner = [tail[0] + offset, tail[1] + offset]
            else:
                rCorner = [head[0] + offset, head[1] - offset]
                lCorner = [tail[0] - offset, tail[1] + offset]

            # Copy values for the corners
            bBox["lCorner"] = lCorner
            bBox["rCorner"] = rCorner

        else:
            bBox["lCorner"] = [0, 0]
            bBox["rCorner"] = [0, 0]
            print("No bounding box")

        return bBox

    def drawBoundingBox(self, image, border = 10):
        """
        Draws bounding box around the given points
        :param image: The image is given to select the bbox
        :return: returns the image with bbox points
        """
        height, width, channel = image.shape
        bBox = {}
        bBox = self.computeBoundingBox()
        # if head is left of the tail
        lCorner = bBox["lCorner"]
        rCorner = bBox["rCorner"]

        if imageOp.isPointValid(lCorner, height, width) and imageOp.isPointValid(rCorner, height, width):
                cv.rectangle(image,(int(lCorner[0]),int(lCorner[1])),(int(rCorner[0]),int(rCorner[1])),(255,0,0),5)
        else :
            print("Bounding box points are not valid")

        return True


    def drawFeatures(self, image, pointSize = 2):
        """
        Drawing feature points using the feature dict
        :param image:
        :return:
        """
        # import ipdb; ipdb.set_trace()
        #pointList = list(self.featureDict.values())
        assert (pointSize > 0)," Error in point size"
        # HYR =(Height, Y coordinate, Rows), WXC = (Width, X coordinate, Cols)
        height, width, channel = image.shape
        for feature in self.featureDict:
            if imageOp.isPointValid(self.featureDict[feature],height,width):
                featureColor = drawOp.getColor(feature)
                drawOp.drawPoint(image,self.featureDict[feature], pointSize, featureColor )
            else:
                drawOp.drawPoint(image,[0,0],8)

    def clearFeatures(self):
        """
        Clear all features
        :return: True
        """
        self.featureDict = {}
        return True

    def drawFeaturePointsOnImg(self, image, pointList, pointSize):
        """
        Drawing feature points using the feature dict
        :param image:
        :return:
        """
        assert (len(pointList) >0)," No points given to draw"
        # HYR =(Height, Y coordinate, Rows), WXC = (Width, X coordinate, Cols)
        height, width, channel = image.shape
        for point in pointList:
            if imageOp.isPointValid(point,height,width):
                drawOp.drawPoint(image, point, pointSize)
            else:
                drawOp.drawPoint(image,[0,0],8)

    def undistortImage(self, image):
        """
        Undistort the image send to the class
        :return: Matrix (Image)
        """
        undistortedImage = cv.undistort(image, self.intrinsicMatrix, self.distortionMatrix)
        if( image.all() == undistortedImage.all()):
            print("Image did not changed after undistortion")

        return undistortedImage

    def projectFeaturesFromCamSpaceToImageSpace(self, camFeaturesDict):
        """
        The function transfers given points from 3D space (in cam space) to 2D space of image
        :param camFeaturesDict: Dictionary of 3D features to be transferred to 2D
        :return: dictionary of transferred 2D points
        """
        pointListCameraSpace = list(camFeaturesDict.values())
        imgPoints = imageOp.projectPointCamSpaceToImgSpace(pointListCameraSpace, self.intrinsicMatrix,
                                                           self.distortionMatrix)
        # Dict : {"Feature":[X,Y,Z]}
        featureList = list(camFeaturesDict)
        imgFeaturesDict = {}
        for index in range(len(featureList)):
            imgFeaturesDict[featureList[index]] = imgPoints[index]

        return imgFeaturesDict

    def projectFeaturesFromViconSpacetoImageSpace(self,ViconFeaturesDict, Rotation,Translation):
        """
        Author: Alex Chan
        Function reprojects points from vicon space directly to camera space from rotation
        translation info
        NOT COMPELTE
        """
        Points3D = list(ViconFeaturesDict.values())
        Allimgpts, jac = cv.projectPoints(Points3D, Rotation, Translation, cameraMatrix, distCoeffs)



    def projectOnImageFromCamSpace(self, pointListCameraSpace):
        """
        projects given 3D points on image space
        :param pointListCameraSpace:
        :return: list of 2D points
        """
        imgPoints = imageOp.projectPointCamSpaceToImgSpace(pointListCameraSpace, self.intrinsicMatrix, self.distortionMatrix)
        return imgPoints

    def projectionMatrix(self, rotationMatrix = np.identity(3), translationMatrix = np.matrix([0,0,0]).T):
        """
        Computes projection matrix from given rotation and translation matrices
        :param rotationMatrix: 3x3 matrix (Default Identity)
        :param translationMatrix: 3x3 Matrix (Default 0)
        :return: 3x4 projection matrix
        """
        projectionMatrix = transformationOp.computeProjectMatrix(self.intrinsicMatrix, rotationMatrix, translationMatrix)
        return projectionMatrix

    def readFeaturePointsFromFile(self, path):
        """
        Loads features from the given path
        :param path:
        :return:
        """
        if os.path.exists(path):
            file = open(path)
            lines = [line.rstrip('\n') for line in file]
            points = []
            # Read point information from the file
            for line in lines:
                self.featureDict[str(line)] = [0,0]

            return True

    def setFeatures(self, featuresDict):
        """
        Set the given features in the feature list
        :param  set the features in the dict using given dict
        :return: True/False
        """
        for feature in featuresDict:
            self.featureDict[feature] = featuresDict[feature]

        # self.generateFeatureMat()
        return True

# TODO : Unit testing of the class and feature saving and management algorithm
if __name__ == "__main__":

    #Old approach
    cam1Points = np.array([[1415.76135949, - 570.12878565, 7724.00698054],
                           [1427.61481598, - 588.07764951, 7708.65018457],
                           [1433.03420332, - 579.43874105, 7728.08209005],
                           [1439.77318296, - 598.69765411, 7698.82965911]])

    distortionCam1 = np.array([[4.25332001e-08, - 1.08721554e-14, 8.49506484e-21, 0.00000000e+00,0.00000000e+00]])

    instrinsicMatCam1 = np.array([[2.64432999e+03, 0.00000000e+00, 9.03959271e+02],
                                  [0.00000000e+00, 2.64432999e+03, 5.47311174e+02],
                                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    imageObject = ImageVicon(0, np.matrix(distortionCam1), np.matrix(instrinsicMatCam1))
    imageObject.readFeaturePointsFromFile("D:\BirdTrackingProject\VICON_DataRead\customFeatures.txt")

    cam1FeatureDict = {"head_beak":[1415.76135949, - 570.12878565, 7724.00698054],
                       "head_eyes":[1427.61481598, - 588.07764951, 7708.65018457],
                       "head_nose":[1433.03420332, - 579.43874105, 7728.08209005],
                       "body_tail":[1439.77318296, - 598.69765411, 7698.82965911]}

    imgPoints = imageObject.projectOnImageFromCamSpace(cam1Points.tolist()) # old
    print("imgPoints" , imgPoints)

    imageFeatures = imageObject.projectFeaturesFromCamSpaceToImageSpace(cam1FeatureDict)
    print("imgPoints Dict", imageFeatures)
    imageObject.setFeatures(imageFeatures)
    img = cv.imread("D:\\BirdTrackingProject\\VICON_DataRead\\bird_rainbow.jpg",cv.IMREAD_COLOR)
    imageObject.drawFeaturePoints(img,1)

    while True:
        cv.imshow("testImg",img)
        k = cv.waitKey(10)
        if k == ord("q"):
            cv.destroyAllWindows()
            break




