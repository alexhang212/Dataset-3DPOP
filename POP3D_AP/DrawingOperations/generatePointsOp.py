# Generate points for drawing operations

def getCoordinatePoints(offset=100):
    """
    Provides point of origin and point on XYZ axis at given offset, to draw coordinate system.
    :param offset: point offset
    :return: List of 3D points [Origin,X,Y,Z]
    """
    og = [0, 0, 0]
    xAxis = [offset, 0, 0]
    yAxis = [0 ,offset, 0]
    zAxis = [0, 0, offset]
    points = [og, xAxis,yAxis,zAxis]
    return points


def getBoundingBox(point3D, xOffset=200, yOffset=100, zOffset=150):
    """
    Creates a 3D bounding box around the given object. Given point and offset values the function returns array of 8 points
    corners of bounding box.
    :param point3D: Point central to bounding box
    :param xOffset: xOffset for bbox
    :param yOffset: yOffset for the bbox
    :param zOffset: zOffset for bbox
    :return: 8 points, corners of cube around the given point with given offset.
    """
    bBoxPoints = []

    #Format = Classic Binary 000,001,010,011,100,101,110,111.
    bBoxPoints.append([point3D[0] + (xOffset * -1), point3D[1] + (yOffset * 1), point3D[2] + (zOffset * 1)])  # 1
    bBoxPoints.append([point3D[0] + (xOffset * 1), point3D[1] + (yOffset * 1), point3D[2] + (zOffset * 1)])  # 2
    bBoxPoints.append([point3D[0] + (xOffset * 1), point3D[1] + (yOffset * -1), point3D[2] + (zOffset * 1)])  # 3
    bBoxPoints.append([point3D[0] + (xOffset * -1), point3D[1] + (yOffset * -1), point3D[2] + (zOffset * 1)])  # 4

    bBoxPoints.append([point3D[0] + (xOffset * -1), point3D[1] + (yOffset * 1), point3D[2] + (zOffset * -1)])  # 5
    bBoxPoints.append([point3D[0] + (xOffset * 1), point3D[1] + (yOffset * 1), point3D[2] + (zOffset * -1)])  # 6
    bBoxPoints.append([point3D[0] + (xOffset * 1), point3D[1] + (yOffset * -1), point3D[2] + (zOffset * -1)])  # 7
    bBoxPoints.append([point3D[0] + (xOffset * -1), point3D[1] + (yOffset * -1), point3D[2] + (zOffset * -1)])  # 8

    return bBoxPoints

def getMarkerPointsDefault():
    """
    Generates marker pattern points
    :return: point pattern array
    """
    pt = [0, 0, 0]
    pt1 = [0, 0, -77.0745]
    pt2 = [0, 13.5987, -29.7795]
    pt3 = [-0.457005, 14.6571, -55.9585]
    pointPattern = [pt, pt1, pt2, pt3]

    return pointPattern

def getMarkerPoints(path):
    file = open(path)
    lines = [line.rstrip('\n') for line in file]
    points = []
    pointPattern = []

    # Read point information from the file
    for line in lines:
        pointInfo = line.split("=")
        if(len(pointInfo) == 2):
            points.append(float(pointInfo[1]))
    # Store pattern information as list
    for i in range(0,len(points),3):
        pt = [points[i],points[i+1],points[i+2]]
        pointPattern.append(pt)

    return pointPattern

def main():
    print("generatePointsOp : VICONDrawingOperations")

    pointPatternFile = getMarkerPoints("9mm_02.mp")
    print("Marker Pattern From File", pointPatternFile)

    pointPattern = getMarkerPointsDefault()
    print("Marker Pattern", pointPattern)

    origin = [0,0,0]
    bBoxPoints = getBoundingBox(origin,100,100,100)
    print("BBoxPoints" , bBoxPoints)

    offset = 10
    coordinateAxis = getCoordinatePoints(offset)
    print("Axis Points", coordinateAxis)


if __name__ == '__main__':
    main()