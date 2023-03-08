# This class is used to provide
import numpy as np
import pyquaternion as pq

def loadDefaultCameraParam():
    """Loading fixed camera parameters for the video camera """
    # 2118670 Camera ID VICON VUE
    f = 2639.60557825127
    px,py = [1010.59313964844 ,549.778747558594]
    distParam = [3.5726620792918736e-008 ,5.7610511811379948e-015 ,-1.7342980478105665e-021, 0]
    intrinsicMatrix = np.zeros((3,3))
    intrinsicMatrix[0,0] =   f
    intrinsicMatrix[1,1] = f
    intrinsicMatrix[0,2] = px
    intrinsicMatrix[1,2] = py
    intrinsicMatrix[2,2] = 1
    return intrinsicMatrix, np.matrix(distParam)

# Load camera extrinsics
def loadDefaultCameraExtrinsics():
    """Loads the extrinsic (C to W) and intrinsic parameters of the camera
    @output : Rotation (4,1) Quaternion, Translation(3,1)"""
    rvec = [0.602704205769858, -0.483347914896162 ,0.398121974542307 ,0.494592081314888]
    tvec = [-4962.51350088183, -2051.0547916964 ,1847.17477466481]
    return rvec, tvec

def quaternionFromVector(vector):
    """Convert to quaternion object from vector form. Format : (x,y,z,w) , w is real, (x,y,z) are imaginary"""
    if len(vector)!=4:
        print("The voctor size wrong")
        raise
    #Format is Quaternion(w,x,y,z)
    quaternion = pq.Quaternion(vector[3],vector[0],vector[1],vector[2])
    return quaternion

class Camera:

    def __init__(self):
        # Init intrinsics
        # Floating point values are important for opencv based image projections
        self.setIntrinsicParam(np.identity(3),np.zeros((1,5)) )
        #Init Camera params
        self.setCameraInfo('NoCam',666)

        # Init extrinsics
        self.setExtrinsicParam([0, 0, 0, 1],[0, 0, 0])


    def setIntrinsicParam(self, intrinsic, distortionParam):
        if (3, 3) == intrinsic.shape:
            self.intrinsicParam = intrinsic
        else:
            raise ValueError("Shape of given matrix is not (3,3).")

        self.distortionParam = distortionParam

    def setCameraInfo(self, cameraType, cameraId):
        if type(cameraType) is str:
            self.cameraType = cameraType
        else:
            raise ValueError("Given camera type not a string value")

        if type(cameraId) is int:
            self.cameraID = cameraId
        else:
            raise ValueError("Given camera ID not an integer value")


    def setExtrinsicParam(self, rot, trans):
        # Shape check for extrinsic rotation and translation
        if len(rot) != 4 or len(trans) != 3:
            raise ValueError("Size of given rot and translation param is wrong.")

        self.extrinsicRotation = rot
        self.extrinsicRotationQuat = quaternionFromVector(self.extrinsicRotation)
        self.extrinsicRotationMatrix = self.extrinsicRotationQuat.rotation_matrix
        self.extrinsicTranslation = trans

        # Possible location to read the camera details from
        # XML Reading capacity of the function, should pluck out camera
    def printCameraParam(self):
        print("Calibration Data : \n")

        print("Camera Type : ", self.cameraType)
        print("Camera ID : ", self.cameraID)

        print("Intrinsic Matrix : ", self.intrinsicParam)
        print("Distortion Param : ", self.distortionParam)

        print("Extrinsic Rotation : ", self.extrinsicRotation)
        print("Extrinsic Quat : ", self.extrinsicRotationQuat)
        print("Extrinsic Translation : ", self.extrinsicTranslation)

    def getCameraParam(self):
        cameraParam = {"intrinsicParam": self.intrinsicParam,
                       "extrinsicRotation": self.extrinsicRotation,
                       "extrinsicTranslation": self.extrinsicTranslation,
                       "distortionParam": self.distortionParam,
                       "serialNo": self.cameraID,
                       "cameraType": self.cameraType}
        return cameraParam


if __name__ == '__main__':
    camObject = Camera()
    camObject.printCameraParam()

    s =  camObject.getCameraParam()
    print("intrinsicParam : " , s["intrinsicParam"])
    print("extrinsicRotation : ", s["extrinsicRotation"])
    print("extrinsicTranslation : ", s["extrinsicTranslation"])
    print("distortionParam : ", s["distortionParam"])


