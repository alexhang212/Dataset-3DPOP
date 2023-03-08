#!/usr/bin/python

from numpy import *
from math import sqrt
from Math import transformations

# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def findPoseFromPoints(pointsInCurrentFrameA, pointsInTargetFrameB,Name=None):
    """
    Function finds pose to transfer points from frame of reference A to frame of reference B.
    :param pointsInCurrentFrameA: 3xN Matrix of point set A
    :param pointsInTargetFrameB: 3xN Matrix of point set B
    :return: 3x3 Rotation Matrix and 3x1 Translation matrix,
    """

    pointsInCurrentFrameA = mat(pointsInCurrentFrameA)
    pointsInTargetFrameB = mat(pointsInTargetFrameB)

    assert len(pointsInCurrentFrameA) == len(pointsInTargetFrameB)

    num_rows, num_cols = pointsInCurrentFrameA.shape

    if num_rows != 3:
        raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))

    [num_rows, num_cols] = pointsInTargetFrameB.shape
    if num_rows != 3:
        raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

    # find mean column wise
    centroid_A = mean(pointsInCurrentFrameA, axis=1)
    centroid_B = mean(pointsInTargetFrameB, axis=1)

    if centroid_A.shape != (3,1) and centroid_B.shape != (3,1):
        centroid_A = centroid_A.reshape(3,1)
        centroid_B = centroid_B.reshape(3,1)

    # subtract mean
    Am = pointsInCurrentFrameA - tile(centroid_A, (1, num_cols))
    Bm = pointsInTargetFrameB - tile(centroid_B, (1, num_cols))

    # dot is matrix multiplication for array
    H = Am * transpose(Bm)

    # find rotation
    U, S, Vt = linalg.svd(H)
    R = Vt.T * U.T

    # special reflection case
    if linalg.det(R) < 0:
        # TEMP
        # if Name == "485_1307_hd":
        #     print("hi")
        # else:
        print("det(R) < R, reflection detected!, correcting for it ...\n");
        Vt[2,:] *= -1
        R = Vt.T * U.T

    t = -R*centroid_A + centroid_B

    return R, t

if __name__ == '__main__':
    test = True

    if test :
        # Test with random data
        p1 = [1009.72, -2437.41, 266.968]
        p2 = [1031.35, -2438.78,267.143]
        p3 = [997.954, -2416.31, 251.854]
        p4 = [1012.95, -2417.49, 264.574]
        point1 = array([p1, p2, p3, p4 ]).T #, p4]
        k1 = [0,0,0]
        k2 = [-7.21645e-16, 3.88578e-16 ,-34.429]
        k3 = [-4.44089e-16, 5.12623,-13.6335]
        k4 = [9.15375, -8.54, -20.903]
        point2 = array([k1, k2, k3, k4]).T # k4]
        # Error is around 13 with this data

        rot, trans = findPoseFromPoints(point1, point2)
        print(" Rotation{0} and Translation{1} : ".format(rot,trans))
        testPoint = dot(rot,mat(p4).T) + trans
        print("Test Point:", testPoint )

        transformedPoints = transformations.transformPoints(point1,rot,trans)
        error = transformations.rmsError(point2,transformedPoints)
        print("Error:", error)


    else :
    # # Random rotation and translation

        random.seed(100)
        R = mat(random.rand(3,3))
        t = mat(random.rand(3,1))

        # make R a proper rotation matrix, force orthonormal
        U, S, Vt = linalg.svd(R)
        R = U*Vt

        # remove reflection
        if linalg.det(R) < 0:
           Vt[2,:] *= -1
           R = U*Vt

        # number of points
        n = 4

        A = mat(random.rand(3, n));
        B = R*A + tile(t, (1, n))

        # Recover R and t
        ret_R, ret_t = findPoseFromPoints(A, B)

        # Compare the recovered R and t with the original
        B2 = (ret_R*A) + tile(ret_t, (1, n))

        # Find the root mean squared error
        err = B2 - B
        err = multiply(err, err)
        err = sum(err)
        rmse = sqrt(err/n)

        print("Points A")
        print(A)
        print("")

        print("Points B")
        print(B)
        print("")

        print("Ground truth rotation")
        print(R)

        print("Recovered rotation")
        print(ret_R)
        print("")

        print("Ground truth translation")
        print(t)

        print("Recovered translation")
        print(ret_t)
        print("")

        print("RMSE:", rmse)

        if rmse < 1e-5:
            print("Everything looks good!\n");
        else:
            print("Hmm something doesn't look right ...\n");