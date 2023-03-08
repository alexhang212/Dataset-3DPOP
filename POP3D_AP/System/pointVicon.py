import numpy as np


class Point3D:
    """
    class creates a 3D feature with name and coordinate information
    """
    def __init__(self, point, name):
        if len(point) != 3:
            raise ValueError(" Need 3 parameters for Point3D class")

        if len(name) == 0:
            raise ValueError(" No string supplied for name of 3D feature. ")

        self.x = point[0]
        self.y = point[1]
        self.z = point[2]
        self.feature = name

    def pointName(self):
        return self.feature

    def pointMat(self):
        point = self.pointList()
        point = np.matrix(point).transpose() # Always 3xN
        return point

    def pointList(self):
        return [self.x, self.y, self.z]

    def setPoint(self, point):
        if len(point) !=0 :
            self.x = point[0]
            self.y = point[1]
            self.z = point[2]
            return True
        else:
            return False

class Feature2D:
    """
    The class generates a 2D feature and stores information about the coordinates with the name
    """
    def __init__(self, point, name):

        if len(point) != 2:
            raise ValueError(" Need 2 parameters for Point2D class")

        if len(name) == 0:
            raise ValueError(" No string supplied for name of 2D feature. ")

        self.x = point[0]
        self.y = point[1]
        self.featureName = name

    def pointMat(self):
        """
        converts the point to matrix 2x1 using point
        :return: martix (2x1)
        """
        point = self.pointList()
        pointMat = np.matrix(point).transpose()
        return pointMat

    def pointList(self):
        """
        returns the point as list
        :return:
        """
        return [self.x, self.y]

    def setPoint(self, point):
        """
        Change coordinates without changing name of the feature
        :param point: List (x,y) coordinates
        :return: True/False
        """
        if len(point) !=0 :
            self.x = point[0]
            self.y = point[1]
            return True
        else:
            return False


################ Unit Test #####################
def checkPoint3D():
    x = [3, 4, 5]
    c = Point3D(x, "team")
    print(" Point as list ", c.pointList())
    print(" Point as mat", c.pointMat(), " Shape : ", c.pointMat().shape)
    print(" Feature Name", c.feature)
    return c

def checkPoint2D():
    x = [3, 4]
    c = Feature2D(x, "team")
    print(" Point as list ", c.pointList())
    print(" Point as mat", c.pointMat(), " Shape : ", c.pointMat().shape)
    print(" Feature Name", c.featureName)
    return c

if __name__ == '__main__':
    point_3d = checkPoint3D()
    point_2d = checkPoint2D()