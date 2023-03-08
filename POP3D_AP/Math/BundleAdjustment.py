"""
Author: Alex Chan
Functions to do bundle adjustment using scipy
Based on this online tutorial: https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

"""

import scipy as sc
import urllib
import bz2
import os
import numpy as np
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import cv2 as cv

def Reproject_AllParams(Points3DAll, CamParamsAll,CamIndexArr,IntMat,Distortion):
    """ 
    Reproject points from 3d to 2d for bundle adjustement of all calibration params (ext + intrinsic)
    """
    AllPoint2D = []
    for i in range(len(Points3DAll)):
        rvec = CamParamsAll[i][0:3]
        tvec = CamParamsAll[i][3:6]
        FundMat = IntMat[CamIndexArr[i]]
        Dist = Distortion[CamIndexArr[i]]

        Point2D = cv.projectPoints(Points3DAll[i], rvec,tvec, FundMat,Dist)
        AllPoint2D.append(Point2D[0][0][0])

    AllPoint2DArr = np.array(AllPoint2D)
    return AllPoint2DArr
    


def fun(params, n_cameras, n_points,CamIndexArr, PointIndexArr, Points2DArr,IntMat,Distortion):
    """
    Compute residuals. Using opencv reproject

    """
    CamParams = params[:n_cameras * 6].reshape((n_cameras, 6))
    Points3D = params[n_cameras * 6:].reshape((n_points, 3))
    points_proj = Reproject_AllParams(Points3D[PointIndexArr], CamParams[CamIndexArr],CamIndexArr,IntMat,Distortion)

    return (points_proj - Points2DArr).ravel()

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

    return A


def getFinalParams(params, n_cameras,n_points):
    """
    Retrieve camera parameters for all params
    """
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    Points3D = params[n_cameras * 6:].reshape((n_points,3))

    return camera_params, Points3D

def BundleAdjust(Points3DArr,Points2DArr,PointIndexArr,CamIndexArr,CamParamArr,Distortion,IntMat):

    n_cameras = CamParamArr.shape[0] 
    n_points = Points3DArr.shape[0]

    n = 6 * n_cameras + 3*n_points#total number of parameters
    m = 2 * Points2DArr.shape[0] #total number of residuals

    x0 = np.hstack((CamParamArr.ravel(), Points3DArr.ravel()))
    f0 = fun(x0, n_cameras, n_points,CamIndexArr, PointIndexArr, Points2DArr,IntMat,Distortion)
    # plt.plot(f0)
    # plt.show()
    A = bundle_adjustment_sparsity(n_cameras, n_points, CamIndexArr, PointIndexArr)
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-8, method='trf',
                args=(n_cameras, n_points,CamIndexArr, PointIndexArr, Points2DArr,IntMat,Distortion))
    # plt.plot(res.fun)
    # plt.show()

    FinalParam, Final3DPoint = getFinalParams(res.x, n_cameras,n_points)

    return Final3DPoint


##########################################
#### Below is code for bundle adjustment, but params all fixed###
#####################################

def Reproject(Points3DAll, CamIndexArr,IntMat,Distortion, Rotation,Translation):
    """ 
    Reproject points from 3d to 2d for bundle adjustement of all calibration params (ext + intrinsic)
    """
    # import ipdb; ipdb.set_trace()
    AllPoint2D = []
    for i in range(len(Points3DAll)):
        rvec = Rotation[CamIndexArr[i]]
        tvec = np.array(Translation[CamIndexArr[i]])
        FundMat = IntMat[CamIndexArr[i]]
        Dist = Distortion[CamIndexArr[i]]

        Point2D = cv.projectPoints(Points3DAll[i], rvec,tvec, FundMat,Dist)
        AllPoint2D.append(Point2D[0][0][0])

    AllPoint2DArr = np.array(AllPoint2D)
    return AllPoint2DArr
    
def Allfun(params, n_cameras, n_points,CamIndexArr, PointIndexArr, Points2DArr,IntMat,Distortion,Rotation,Translation):
    """
    Compute residuals. Using opencv reproject

    """
    Points3D = params.reshape((n_points, 3))
    # import ipdb;ipdb.set_trace()
    points_proj = Reproject(Points3D[PointIndexArr],CamIndexArr,IntMat,Distortion,Rotation,Translation)

    return (points_proj - Points2DArr).ravel()

def bundle_adjustment_sparsity_Fix(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)

    for s in range(3):
        A[2 * i,  point_indices * 3 + s] = 1
        A[2 * i + 1,point_indices * 3 + s] = 1

    return A

def getFinalParams3D(params,n_points):
    """
    Retrieve camera parameters for all params with all params fixed, 3d points only
    """
    Points3D = params.reshape((n_points,3))

    return Points3D


def BundleAdjustFixAll(Points3DArr,Points2DArr,PointIndexArr,CamIndexArr,CamParamArr,Distortion,IntMat,Rotation,Translation):
    """ Bundle adjustment of all 3D points, all parameters fixed"""
    n_cameras = CamParamArr.shape[0] 
    n_points = Points3DArr.shape[0]
    # import ipdb;ipdb.set_trace()

    n = 3*n_points#total number of parameters
    m = 2 * Points2DArr.shape[0] #total number of residuals

    x0 = Points3DArr.ravel()
    f0 = Allfun(x0, n_cameras, n_points,CamIndexArr, PointIndexArr, Points2DArr,IntMat,Distortion,Rotation,Translation)
    # plt.plot(f0)
    # plt.show()
    A = bundle_adjustment_sparsity_Fix(n_cameras, n_points, CamIndexArr, PointIndexArr)
    res = least_squares(Allfun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-8, method='trf',
                args=(n_cameras, n_points,CamIndexArr, PointIndexArr, Points2DArr,IntMat,Distortion,Rotation,Translation))
    # plt.plot(res.fun)
    # plt.show()

    Final3DPoint = getFinalParams3D(res.x,n_points)

    return Final3DPoint



