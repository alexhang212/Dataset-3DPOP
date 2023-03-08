# The file generates read write functions to load vicon calibration file
# Currently only supported for custom class for RGB cameras, in future might add support for infrared camera
# Import .xml file and read the calibration parameters

from xml.dom import minidom
from System import camera as cam
import numpy as np

def readCalibrationFromXCP(calibFile):
    """ The input is .xml/.xcp file and output is objects of calibration data """
    xmlDoc = minidom.parse(calibFile)
    cameraList = xmlDoc.getElementsByTagName('Camera')

    noOfVueCam = 0
    noOfVeroCam = 0
    camObjects = []
    for camera in cameraList:
        camType = camera.attributes['DISPLAY_TYPE'].value
        if camType == "Vue": # If camera is vue = RGB save the calibration parameters
            noOfVueCam += 1
            cameraObject = cam.Camera()

            x = camera.getElementsByTagName('KeyFrame')
            if len(x) == 1: # If Vue camera was activated during calibration len(x) == 1
                f = x[0].attributes['FOCAL_LENGTH'].value
                extrinsicPostion = (x[0].attributes['POSITION'].value).split()
                tvec = [float(extrinsicPostion[0]), float(extrinsicPostion[1]), float(extrinsicPostion[2])]
                extrinsicRotation = (x[0].attributes['ORIENTATION'].value).split()
                rvec = [float(extrinsicRotation[0]), float(extrinsicRotation[1]), float(extrinsicRotation[2]),
                        float(extrinsicRotation[3])]
                principalPoint = (x[0].attributes['PRINCIPAL_POINT'].value).split()
                dist = (x[0].attributes['VICON_RADIAL2'].value).split()

                # Matrix composition
                focal = float(f)
                px = float(principalPoint[0])
                py = float(principalPoint[1])
                #todo : Remove this from the code, or inser 0 values as we use undistorted videos already.
                distParam = [float(dist[3]), float(dist[4]), 0, 0, float(dist[5])]  # Convert string to float

                intrinsicMatrix = np.zeros((3, 3))
                intrinsicMatrix[0, 0] = focal
                intrinsicMatrix[1, 1] = focal
                intrinsicMatrix[0, 2] = px
                intrinsicMatrix[1, 2] = py
                intrinsicMatrix[2, 2] = 1
                distortionMatrix = np.matrix(distParam)

                # Get camera information
                #print('Type : ', camera.attributes['DISPLAY_TYPE'].value)
                #print('ID : ', camera.attributes['DEVICEID'].value)
                cameraObject.setCameraInfo(camera.attributes['DISPLAY_TYPE'].value,
                                           int(camera.attributes['DEVICEID'].value))

                # Use the camera object class and defind the required parameters
                cameraObject.setIntrinsicParam(intrinsicMatrix, distortionMatrix)
                cameraObject.setExtrinsicParam(rvec, tvec)
                camObjects.append(cameraObject)
            else: # If Vue camera was deactivated during calibration len(x) == 0
                pass
        else: # If camera is Vero ( it is infra red camera )
            noOfVeroCam += 1

    print("Total Cameras : ", len(cameraList), " - Vue : ", noOfVueCam, " + Vero : ", noOfVeroCam)
    # Return the camera Object
    return camObjects

if __name__ == '__main__':

    camObjects = readCalibrationFromXCP("D:\\BirdTrackingProject\\20180906_1BirdBackpack\\20180906_1BirdBackpack_Bird201.xcp")
    for camera in camObjects:
        camera.printCameraParam()
        s =  camera.getCameraParam()
        print("intrinsicParam : " , s["intrinsicParam"])
        print("extrinsicRotation : ", s["extrinsicRotation"])
        print("extrinsicTranslation : ", s["extrinsicTranslation"])
        print("distortionParam : ", s["distortionParam"])
        print("Cam ID", s["serialNo"])
        print("Camera Type", s["cameraType"])
