# Transformations contains all basic methods to transform from one coordinate system to another.

import pyquaternion as pq
import math
import numpy as np



# TODO: Remove function it is deprected
def worldToCameraSpace(pointList3D, rotationList, translationList):
    """
    Transform point from one world to camera, for VICON specific setup
    :param pointList3D: List of 3D points
    :param rotationList: List of rotation parameters  (Euler angles or Quaternion)
    :param translationList: translation parameter
    :return: Transformed points
    """
    rotationQ = pq.Quaternion()  # init
    if (len(rotationList) == 3):
        rotationQ = angleToQuaternion(rotationList)
        rotationMatrix = rotationQ.rotation_matrix
    elif (len(rotationList) == 4):
        rotationQ = quaternionFromVectorVICON(rotationList)  # get new quat from function
        rotationMatrix = rotationQ.rotation_matrix
    else:
        print("Input format of rotation incompatible")
        raise ValueError("Error in rotation values")

    # Convert translation matrix 3x1
    translationMatrix = np.matrix(translationList)
    translationMatrix = translationMatrix.transpose()

    # Convert to Matrix and reshape from Nx3 to 3xN
    pointMatrix = np.matrix(pointList3D)
    pointMatrix = pointMatrix.transpose() # 3xN

    # P_camera (3xN) = R (3x3).(P_world (3xN)- T (3x1))(3xN)
    # We have to assume that R given to transfer point from Object space to World Space is already inversed
    # Therefore we perform the steps usually performed to inverse the point but here we do not invert the rotation matrix
    transformedPoints = invertPoints(pointMatrix, rotationMatrix, translationMatrix)
    transformedPoints = transformedPoints.transpose() # converting to Nx3

    return transformedPoints


def invertPoints(pointMatrix, invertedRotationMatrix, translationMatrix):
    """
    inverts the points based on given rotation and translation matrix
    :param pointMatrix : 3xN (matrix)
    :param rotationMatrix : 3x3 rotation Matrix (3x3)
    :param translationMatrix : 3x1 translation matrix
    :return: Inverted point matrix : 3xN (matrix)
    """

    assert( pointMatrix.shape[0] == 3), "Expected dimensions of points are 3XN"
    assert (invertedRotationMatrix.shape == (3, 3) and
            translationMatrix.shape == (3, 1)) , "Invalid dimension of rotation or tranlsation matrix"

    translatedPoints = pointMatrix - translationMatrix
    transformedPoints = np.dot(invertedRotationMatrix, translatedPoints)

    return transformedPoints

def transformPoints(pointMatrix, rotationMatrix, translationMatrix):
    """
    transforms points based on given rotation and translation matrix
    :param pointMatrix : Matrix 3xN
    :param rotationMatrix : rotation matrix 3x3
    :param translationMatrix : translation matrix 3x1
    :return : Matrix 3xN
    """

    assert( pointMatrix.shape[0] == 3), "Expected dimensions of points are 3XN"
    assert (rotationMatrix.shape == (3, 3) and
            translationMatrix.shape == (3, 1)) , "Invalid dimension of rotation or tranlsation matrix"

    # P_out (3xN) = R (3x3) . P (3xN) + T (3x1)
    rotatedPoints = np.dot(rotationMatrix, pointMatrix)
    transformedPoints = np.add(rotatedPoints, translationMatrix)

    return transformedPoints

def rmsErrorPerPoint(X,Y):
    """
    Compute error between set of points
    :param X: 3XN array
    :param Y: 3XN array
    :return: 1xN array
    """
    assert (X.shape == Y.shape), " Shape of given array do not match"
    meanErrorPerPoint = np.sqrt(np.sum(np.square(X - Y), 0))

    return meanErrorPerPoint

def rmsError(X,Y):
    """
    Compute error between set of points
    :param X: 3XN array
    :param Y: 3XN array
    :return: error
    """
    assert (X.shape == Y.shape), " Shape of given array do not match"
    meanError = np.mean(np.sqrt(np.sum(np.square(X - Y), 0)))

    return meanError


def invertQuaternion(rotation):
    """
    inverts the given quaternion and returns as list of rotation
    :param rotation: list (rotation parameters : vicon format )
    :return:  list (rotation param : vicon format x,y,z,w
    """
    quat = quaternionFromVectorVICON(rotation)
    inverseQuat = quat.inverse
    return viconVectorFromQuaternion(inverseQuat)


def computeProjectMatrix(intrinsicMatrix, rotationMatrix, translationMatrix):
    """
    Computes projection matrix from given rotation and translation matrices
    :param rotationMatrix: 3x3 matrix
    :param translationMatrix: 3x3 Matrix
    :return: 3x4 projection matrix
    """
    assert (rotationMatrix.shape == (3, 3) and
            translationMatrix.shape == (3, 1) and
            intrinsicMatrix.shape == (3,3)), "Matrix shape mismatch"

    RT = np.concatenate((rotationMatrix, translationMatrix), axis=1)
    projectionMatrix = np.dot(intrinsicMatrix, RT)

    return projectionMatrix


def transformPointsUsingViconParam(pointList3D, rotationList, translationList, inverse=False):
    """
    Transform point from one 3D space to another using given rotation (Quaternion or Euler Angles) and translation param
    :param pointList3D: list of points Nx3
    :param rotationList: Rotation parameters from vicon (EurlerXYZ with 3 param or Quaternion with 4 param)
    :param translationList: Translation parameters with 3 param
    :param inverse: To invert points. Nx3
    :return: List of transferred points
    """

    assert (len(rotationList) == 3 or len(rotationList) == 4), "Rotation parameters do not match specifications, " \
                                                               "3 for Euler angles or 4 for quaternions (VICON format)."

    transferPoints = np.array(pointList3D)
    assert (transferPoints.shape[0] == 3 or transferPoints.shape[1] == 3), "The given points are not in Nx3 or 3xN format"
    # If given array is Nx3 we convert it to 3xN for standard multiplication
    pointsTransposed = False
    if transferPoints.shape[1] == 3:
        transferPoints = transferPoints.T
        pointsTransposed = True

    rotMatrix, translationMatrix = transformationParamListToMatrix(rotationList, translationList, inverse)

    if inverse:
        # P_out (3xN) = R_inv (3x3) . (P_in) (3xN) - T (3x1))
        transformedPoints = invertPoints(transferPoints, rotMatrix, translationMatrix)

    else:
        # P_out (3xN) = R (3x3) . P (3xN) + T (3x1)
        transformedPoints = transformPoints(transferPoints, rotMatrix, translationMatrix)

    # Convert points from 3xN to Nx3
    if pointsTransposed:
        transformedPoints = transformedPoints.transpose()

    return transformedPoints


# Todo: Refatured the function can use transformPointsUsingViconParam instead (defined above) for elegant implementation
def  transformPoint3D(pointList3D, rotationList, translationList, inverse=False, custom=False):
    """
    Transform point from one 3D space to another using given rotation (Quaternion or Euler Angles) and translation param
    :param pointList3D: list of points Nx3
    :param rotationList: Rotation parameters from vicon (EurlerXYZ with 3 param or Quaternion with 4 param)
    :param translationList: Translation parameters with 3 param
    :param inverse: To invert points. Nx3
    :return: List of transferred points
    """
    # import ipdb; ipdb.set_trace()


    # Initialize quaternion
    rotationQ = pq.Quaternion()
    rotationMatrix = []
    if custom:
        rotationQ = pq.Quaternion(rotationList[0], rotationList[1], rotationList[2], rotationList[3])
        rotationMatrix = rotationQ.rotation_matrix

    elif (len(rotationList) == 3):
        # print("Angle to Quat ")
        rotationQ = angleToQuaternion(rotationList)
        rotationMatrix = rotationQ.rotation_matrix
    elif (len(rotationList) == 4):
        # print("Vector to Quat")
        rotationQ = quaternionFromVectorVICON(rotationList)  # get new quat from function
        rotationMatrix = rotationQ.rotation_matrix
    else:
        print("Input format of rotation incompatible")

    rotationInv = rotationQ.inverse
    rotationMatrixInv = rotationInv.rotation_matrix

    # Convert translation matrix 3x1
    translationMatrix = np.matrix(translationList)
    translationMatrix = translationMatrix.transpose()

    # Convert to Matrix and reshape from Nx3 to 3xN
    transferPoints = np.matrix(pointList3D)
    transferPoints = transferPoints.transpose()

    if inverse:
        # P_out (3xN) = R_inv (3x3) . (P_in) (3xN) - T (3x1))
        transformedPoints = invertPoints(transferPoints, rotationMatrixInv, translationMatrix)

    else:
        # P_out (3xN) = R (3x3) . P (3xN) + T (3x1)
        transformedPoints = transformPoints(transferPoints, rotationMatrix, translationMatrix)

    # Convert points from 3xN to Nx3
    transformedPoints = transformedPoints.transpose()

    return transformedPoints

'''Rotation functions : To covert the rotation from one angle to another
    Angles (Degrees) -> Matrix (3x3)
    List -> Quaternion
    Quaterion -> Rotation Matrix (3x3)
    Rotation Matrix -> Quaternion (4x1)
    Euler Angle (Degrees) -> Quaternion
'''

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):
    """
    Converts euler angles to rotation matrix(3x3) = Rx.Ry.Rz (VICON Documentation)
    :param theta: Angle for rortation with Euler Anglers
    :return: Rotation matrix
    """

    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_x, np.dot(R_y, R_z))
    return R

def viconVectorFromQuaternion(quatRotation):
    """
    returns vector format for the given quaternion
    :param quaternion: Quaternion object
    :return: Vector format
    """
    vector = quatRotation.elements # Format from library is : w, x, y, z (w is imaginary)
    # Vicon format for quaternion is x,y,z,w, therefore we convert
    return [vector[1], vector[2], vector[3], vector[0]]

def quaternionFromVectorVICON(vector):
    """
    Converts a list of rotation parameters (from vicon format) to quaternion  : w is real, x,y,z are imaginary
    :param vector: list 4 elements x,y,z,w (vicon Format)
    :return: Quaternion object in pyquaternion package format
    """
    if len(vector) != 4:
        print("The voctor size wrong")
    # Input format according requirement of package  : w, x, y, z
    quaternion = pq.Quaternion(vector[3], vector[0], vector[1], vector[2])
    return quaternion

def translationListToMatrix(transList):
    """
    Convert tranlation list to translation Matrix
    :param transList:
    :return: Matrix (3x1)
    """
    assert(len(transList) == 3), "Size error translation parameter length not 3"
    translationMatrix = np.array(transList)
    translationMatrix = translationMatrix.reshape(3,1)
    return translationMatrix

def eulerListToMatrix(rotList, inversion = False):
    """
    Convert list of euler angler to Matrix
    :param rotList: list of 3 param
    :return: Tuple of two rotation matrix 3x3, one for forward transformation and another for backward
    """
    assert (len(rotList) == 3), "Method expects a list of 3 parameters for quaternions"
    if inversion == False:
        rotationQ = angleToQuaternion(rotList) # Convert to quaternions
        rotationMatrix = rotationQ.rotation_matrix # Convert to rotation matrix
        return rotationMatrix
    else:
        rotationQ = angleToQuaternion(rotList)  # Convert to quaternions
        rotationInv = rotationQ.inverse # Invert the rotation matrix
        rotationMatrixInv = rotationInv.rotation_matrix
        return rotationMatrixInv

def quaternionViconListToMatrix(rotList, inversion = False):
    """
    Convert quaternion to rotation matrix
    :param rotList: List of param
    :param inversion: Bool
    :return: Rotation matrix 3x3, for forward transformation or backward transformation
    """
    assert (len(rotList) == 4), "Method expects a list of 4 parameters for quaternions"

    if inversion == False:
        quat = quaternionFromVectorVICON(rotList)  # create quaternion from list
        rotationMatrix = quat.rotation_matrix
        return rotationMatrix
    else:
        quat = quaternionFromVectorVICON(rotList)
        rotationInv = quat.inverse
        rotationMatrixInv = rotationInv.rotation_matrix
        return rotationMatrixInv



def transformationParamListToMatrix(rotation, translation, inversion = False):
    """
    Return rotation, translation matrix from given rotation list
    :param rotation: list of rotation param
    :param translation: list of translation param
    :return: rotation matrix (3x3), translation matrix (3x1)
    """


    rotationMatrix = []
    if (len(rotation) == 3): # If euler angles
        rotationMatrix = eulerListToMatrix(rotation, inversion)
    elif (len(rotation) == 4): # If quaternion parameters
        rotationMatrix = quaternionViconListToMatrix(rotation, inversion)
    else:
        rotationMatrix = np.eye(3)
        print("Input format of rotation incompatible")

    translationmatrix = translationListToMatrix(translation)

    return rotationMatrix, translationmatrix

def transformationParamListToMatrixCustom(rotation, translation, inversion = False):
    """
    Author: Alex Chan, additional function wrote for custom objects, the way rotation
    was saved is different from vicon
    
    Return rotation, translation matrix from given rotation list
    :param rotation: list of rotation param
    :param translation: list of translation param
    :return: rotation matrix (3x3), translation matrix (3x1)
    """
    # import ipdb; ipdb.set_trace()

    rotationMatrix = []
    if (len(rotation) == 3): # If euler angles
        rotationMatrix = eulerListToMatrix(rotation, inversion)
    elif (len(rotation) == 4): # If quaternion parameters
        rotationQ = pq.Quaternion(rotation[0], rotation[1], rotation[2], rotation[3])
        # rotationInv = rotationQ.inverse
        rotationMatrix = rotationQ.rotation_matrix
    else:
        rotationMatrix = np.eye(3)
        print("Input format of rotation incompatible")

    translationmatrix = translationListToMatrix(translation)

    return rotationMatrix, translationmatrix


def rotMatrixToQuat(rotMatrix):
    """convert rotation matrix to quaternion form"""

    quat = pq.Quaternion(matrix=rotMatrix)
    return quat


def angleToQuaternion(theta):
    """convert quaternion from angles"""
    rotMatrix = eulerAnglesToRotationMatrix(theta)
    rotQuat = pq.Quaternion(matrix=rotMatrix)
    return rotQuat

def CenterCoordinates(RealCoord):
    """Center list of 4 3D points to pt1"""
    CenteredPoints = []
    for point in RealCoord:
        CorrectedPoint = [point[i]-RealCoord[0][i] for i in range(3)]
        CenteredPoints.append(CorrectedPoint)
    
    return CenteredPoints




def unitTest():
    """
    The function does unit test on all the given functions in the file, to test the transformation in better way.
    :return: None
    """
    #Debug Tests for all functions
    print("Printing results for all basic functions used in the file.")


    theta = [0,0,0]
    print(" Test rotation angles:", theta)
    rot = eulerAnglesToRotationMatrix(theta)
    print(" Anglers to Rotation Matrix {}: {} ".format(rot.shape,rot))

    quat = rotMatrixToQuat(rot)
    print("Quaternion from rotation matrix: ", quat)

    quatVec = [quat[0],quat[1],quat[2],quat[3]]
    print("Quat parameters: ", quatVec)

    quatVecVicon = viconVectorFromQuaternion(quat)
    print("Quat vectors in vicon format: ", quatVecVicon)

    rotMatrix = quaternionViconListToMatrix(quatVecVicon)
    print("RotMatrix from quaternion params in vicon format {}: {} ".format(rotMatrix.shape,rotMatrix))

    rotQuat = angleToQuaternion(theta)
    print("Quaternion parameters from rotation angles directly: ", rotQuat)

    t = [3, 3, 1]
    transMax = translationListToMatrix(t)
    print("TransMatrix {}:{}".format(transMax.shape,transMax) )

    rotationMatrix, translationMatrix = transformationParamListToMatrix(quatVecVicon, t)
    print(" Matrix from List : Rotation Matrix: {}, Translation Matrix : {}".format(quatVecVicon,t) )

    rotationMatrixInv, translationMatrix = transformationParamListToMatrix(quatVecVicon, t, inversion= True)
    print(" Inv Matrix from list : Rotation Matrix Inv : {}, Translation Matrix : {}".format(quatVecVicon, t))

    print("Validating inversion result should be Identity: {}", np.dot(rotationMatrix,rotationMatrixInv))

    projectionMatrix = computeProjectMatrix(np.random.randn(3,3),rotationMatrix,translationMatrix)
    print("Projection Matrix", projectionMatrix)
    # Testing the conversion of quaternion rotation from vector list of vicon, vice-versa
    Q = [1, 2.2, 1.3, 3.4]
    quat = quaternionFromVectorVICON(Q)
    print(quat)
    viconvector = viconVectorFromQuaternion(quat)
    print(" Vicon vector : ", viconvector)

    # Testing inversion of the rotation parameters from vicon list format to vicon list format
    inversion = invertQuaternion(Q)
    invQuat = quaternionFromVectorVICON(inversion)
    quat = quaternionFromVectorVICON(Q)
    mul = invQuat * quat
    print(" It should be identity : ", mul.rotation_matrix)

    point3D = [0, 0, 0]
    rotationAngle = [0, 0, 0]
    print(rotationAngle)
    print("OG point", point3D)
    t = [2002.48, -1735.68, 553.519]
    points = []
    points.append(point3D)
    s = transformPointsUsingViconParam(points, rotationAngle, t)
    print(" Transformed point : ", s)
    p = transformPointsUsingViconParam(s, rotationAngle, t, True)
    print("Bringing back the transformed point: ", p)

def unitTestForPointTransformation():

    # Debug tests for functions from real data
    points = [[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]]
    print("Points OG : ", points)
    rotation = [-0.32789600000000002, -0.21135899999999999, 0.76417400000000002, -0.51366299999999998]
    translation = [-77.495999999999995, 908.19100000000003, 144.577]

    # Transform points using rotation parameters , ideally quaternions
    # Old implementation
    transformedPoints = transformPoint3D(points, rotation, translation)
    print("Transformed points ", transformedPoints )
    testPoint = transformPoint3D( transformedPoints , rotation , translation,True)
    print("Inversre Points points ", testPoint )
    error = rmsError(np.array(points), testPoint)
    print("Error with R/t: ", error)

    # Transform points using rotation parameters , ideally quaternions
    # New implementation
    transformedPoints = transformPointsUsingViconParam(points, rotation, translation)
    print("Transformed points ", transformedPoints)
    testPoint = transformPointsUsingViconParam(transformedPoints, rotation, translation,inverse = True)
    print("Inversre Points points ", testPoint)
    # Error between points with transformation
    error = rmsError(np.array(points),testPoint)
    print("Error with R/t: ", error)

# Default functions which is called when the file is called on its own
def main():

    unitTest()
    unitTestForPointTransformation()


if __name__ == '__main__':
    main()