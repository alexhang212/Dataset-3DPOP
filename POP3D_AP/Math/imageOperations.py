# The file contains functions to do image operations
import numpy as np
import cv2 as cv
from System import camera as cam
import math

# todo : Convert the function for processing more points at a time, the current method does it but the output is nested array for no reason.
def projectPointCamSpaceToImgSpace(point3d, k, dist):
    """Projects a list of Nx3 3D points from space to image space N x 2

    keyword arguments:
    point3d -- 3D points Nx3
    k -- intrinsic
    dist -- dist camera distortion parameters 4x1
    return -- image points 2D Nx2

    """
    # Create identity matrix as transformation from world to camera space, openCV projection requirements
    rot = np.identity(3)
    trans = np.float64([[0,0,0]])
    # Converting point to floating point to meet openCV requirements
    point3d = np.matrix( point3d)
    point3d = np.float64(point3d)
    noOfPoints = point3d.shape[0]
    if (noOfPoints == 0):
        print("No points given in list")
        raise

    # project point on the image
    # *Note direction VICON rot and trans are not given because the transformation is different (Check transformations.worldToCameraSpace())
    # import ipdb; ipdb.set_trace()
    imgProjections, jacobian = cv.projectPoints(point3d, rot, trans, k, dist)
    # if np.isnan(imgProjections).all()== False:
    #     import ipdb;ipdb.set_trace()
    # # Create empty list of points to rearrange shape from (N,1,2) -> (Nx2)
    imgPoints = np.matrix(imgProjections)
    imgPoints = imgPoints.tolist()

    return imgPoints

# todo : Write a function that projects points from world space to image space
def prointPointWorldSpaceToImgSpace(points3D, k , dist , rot , trans):
    imgPoints = []
    return imgPoints

def isPointValid(point, rows, cols ):
    """Check if the point is within the given image dimensions"""
    #print("Point", point , "cols: " , cols , "rows : ", rows)
    status = False
    if math.isnan(point[0]) :
        return status
    elif 0 <= int(point[1]) < rows and 0 <= int(point[0]) < cols: # x limit in width, y limit in height
        status = True
    return status

def filterBBoxPoints(points):
    origin =  [0,0]
    distances = []
    for pt in points:
        xDist = pt[0] - origin[0]
        yDist = pt[1] - origin[1]
        dist = (xDist*xDist) + (yDist*yDist)
        dist = np.sqrt(dist)
        distances.append(dist)

    minDist = min(distances)
    maxDist = max(distances)
    minIndex = distances.index(minDist)
    maxIndex = distances.index(maxDist)

    return minIndex,maxIndex

def main():

    points = [[1,1],[22,22],[15,20],[35,34]]
    points = np.array(points)
    min,max = filterBBoxPoints(points)
    print("Min", points[min] )
    print("Max", points[max] )

    camera = cam.Camera()
    param = camera.getCameraParam()
    k = param["intrinsicParam"]
    dist = param["distortionParam"]
    print("Intrinsic : ", k)
    print("dist: ", dist , " - Shape ", dist.shape)
    p1 = [np.random.randint(6000), np.random.randint(500), np.random.randint(3000)]
    p2 = [np.random.randint(6000), np.random.randint(500), np.random.randint(3000)]
    point = [p1,p2]
    point = np.array(point)
    print("Shape 3D points : " , point.shape)
    imgPoint = projectPointCamSpaceToImgSpace(point, k, dist)
    print(" image projection Shape : ", imgPoint.shape)

    for i in range(imgPoint.shape[0]):
        print("Point: ", point[i], " <--> Projection : ", imgPoint[i], " -->", isPointValid(imgPoint[i], 1080, 1920))

if __name__ == '__main__':
    main()