import cv2
import numpy as np
import math


def detectObject( imgBk, img, threshold ):

    difference = cv2.absdiff(imgBk, img)
    differenceS = cv2.split(difference)
    object = cv2.inRange(differenceS[2], threshold, 255)
    
    mask = cv2.inRange(differenceS[0], 50, 255)
    mask = cv2.bitwise_or(object, mask)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations = 1)
    mask = cv2.erode(mask, kernel, iterations = 2)
    return object, mask


def combineImages(img, imgBk, mask):

    locs = np.where(mask != 0)
    res = imgBk
    res[locs[0], locs[1], :] = img[locs[0], locs[1], :]
    return res

def angleBetween(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def angleOf(p1):
    p2 = (0, 1)
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def filterContour(contoursIn):
    return contoursIn

def defineContours(contours, aproximation, numberMinPointsThreshold, numberMaxPointsThreshold):

    contoursResult = []
    for i in range(len(contours)):
        contourAP = cv2.approxPolyDP(contours[i], aproximation, 1)
        if contourAP.size > numberMinPointsThreshold:
            contoursResult.append(contourAP)

    # To add contour filtering
    contoursFiltered = filterContour(contoursResult)

    return contoursResult, contoursFiltered

def boundingBox(contours):

    area = 0
    index = 0
    for i in range(len(contours)):
        areaC = cv2.contourArea(contours[i])
        if areaC > area:
            area = areaC
            x, y, w, h = cv2.boundingRect(contours[i])
            index = i

    return x, y, w, h, index

def redefineObject(object, contour):
    kernel = np.ones((5, 5), np.uint8)

    maskObject = np.zeros((object.shape), np.uint8)
    i = 0
    ratio = 0
    while ratio < 0.5:
        cv2.drawContours(maskObject, [contour[0]], -1, (255,255,255), -1, cv2.LINE_AA)
        maskObject = cv2.erode(maskObject, kernel, iterations = 1)
        object, contour, hierarchy = cv2.findCont1ours(maskObject,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        area = cv2.contourArea(contour[0])
        hull = cv2.convexHull(contour[0])
        hull_area = cv2.contourArea(contour[0])
        ratio = float(area)/hull_area
        print(ratio)
        cv2.imshow("hull", hull)
        cv2.imshow("object", object)
        cv2.imshow("maskObject", maskObject)
        cv2.waitKey(0)

    return maskObject

def features(img, maskObject, contour):

    # Centroid
    M = cv2.moments(maskObject)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    # Area
    area = cv2.contourArea(contour)

    # Perimeter
    perimeter = cv2.arcLength(contour,1)

    # Orientation
    if len(contour) > 5:
        (xE,yE),(ma,MA),angle = cv2.fitEllipse(contour)
        if MA > 0:
            aspectRatio = ma/MA
        else:
            aspectRatio = 1000
    else:
        angle = 0
        aspectRatio = 0

    #Solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area

    # weighted centroid acoording with gray level luminosity
    imgG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imageMask = cv2.bitwise_and(imgG,maskObject)
    x = range(0, imageMask.shape[0])
    y = range(0, imageMask.shape[1])

    (X,Y) = np.meshgrid(x,y)
    X = np.transpose(X)
    Y = np.transpose(Y)

    x_coord = (X*imageMask).sum() / imageMask.sum().astype("float")
    y_coord = (Y*imageMask).sum() / imageMask.sum().astype("float")

    #Mean Color
    mean_val = cv2.mean(img, mask = maskObject)

    area = M['m00']/255 # area expressed as momentum order 0 is accurate

    vector = (cx - y_coord, cy - x_coord)
    angleIllumination = angleOf(vector)
    distanceIllumination = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1])

    return [cx, cy, area, perimeter, angle, aspectRatio, solidity, int(y_coord), int(x_coord), mean_val[0], mean_val[1], mean_val[2], int(ma/2), int(MA/2), int(xE), int(yE), angleIllumination, distanceIllumination]


def saveFileXML(filename, resultFolder, path, heigth, width, channels, name, x, y, w, h):

    pos = filename.find(".")
    xmlName = path + resultFolder + "/" + filename[:pos] + ".xml"
    xmlFile = open (xmlName, 'w')
    xmlFile.write ("<annotation>" + "\n")
    xmlFile.write ("\t" + "<folder>" + resultFolder + "</folder>" + "\n")
    xmlFile.write ("\t" + "<filename>" + filename + "</filename>" + "\n")
    xmlFile.write ("\t" + "<path>" + path + "/" + filename + "</path>\n")
    xmlFile.write ("\t" + "<source>\n\t\t<database>Unknown</database>\n\t</source>\n")
    xmlFile.write ("\t<size>\n\t\t<width>" + str(width) + "</width>\n\t\t<height>" + str(heigth) + "</height>\n\t\t<depth>" + str(channels) + "</depth>\n\t</size>\n")
    xmlFile.write ("\t<segmented>0</segmented>\n\t<object>\n")
    xmlFile.write ("\t\t<name>" + name + "</name>\n")
    xmlFile.write ("\t\t<pose>Unspecified</pose>\n\t\t<truncated>0</truncated>\n\t\t<difficult>0</difficult>\n")
    xmlFile.write ("\t\t<bndbox>\n\t\t\t<xmin>" + str(x) + "</xmin>\n")
    xmlFile.write ("\t\t\t<ymin>" + str(y) + "</ymin>\n")
    xmlFile.write ("\t\t\t<xmax>" + str(x + w) + "</xmax>\n")
    xmlFile.write ("\t\t\t<ymax>" + str(y + h) + "</ymax>\n")
    xmlFile.write ("\t\t</bndbox>\n\t</object>\n")
    xmlFile.write ("</annotation>")

    xmlFile.close()
    return


def plotData(dataX, dataY, axisXRange = [], axisYRange = [], color = (128,255,128), base =[], steps = [], type = "line", title = "",  axisColor = (255,255,255)):

    if base==[]:
        base = np.zeros((480,640,3), np.uint8)

    h, w, _ = base.shape

    if axisXRange==[]:
        axisXRange=[np.amin(dataX), np.amax(dataX)]

    if axisYRange==[]:
        axisYRange=[np.amin(dataY), np.amax(dataY)]

    axisRange = np.append(axisXRange, axisYRange)

    if steps==[]:
        steps = [int(round(w/100)), int(round(h/100))]

    font = cv2.FONT_HERSHEY_SIMPLEX
    xMargin = 45
    yMargin = 25
    y1Margin = 10

    xMin = axisRange[0]
    xMax = axisRange[1]
    yMin = axisRange[2]
    yMax = axisRange[3]

    scaleX = (w-(2*xMargin))/(xMax-xMin)
    scaleY = (h-(2*(yMargin+y1Margin)))/(yMax-yMin)
    xStep = steps[0]
    xCoor = np.arange(xMargin,w-xMargin,int((w- xMargin - xMargin)/xStep))
    xCoor = np.append(xCoor, w-xMargin)
    inc = ((xMax-xMin)/xStep)

    xValues = np.arange(float(xMin),float(xMax),inc)
    xValues = np.append(xValues, xMax)
    yStep = steps[1]
    yCoor = np.arange(yMargin + y1Margin,h-yMargin-y1Margin,int((h- 2*(yMargin + y1Margin))/yStep))
    yCoor = np.append(yCoor, h-yMargin-y1Margin)

    inc = ((yMax-yMin)/yStep)

    yValues = np.arange(float(yMax),float(yMin),-inc)
    yValues = np.append(yValues, yMin)
    for xC,xV in zip(xCoor, xValues):
        cv2.putText(base,str(round(xV,2)),(xC-10 ,h-yMargin+5), font, 0.4, axisColor,1,cv2.LINE_AA)
        cv2.line(base, (xC ,h-yMargin-y1Margin+4), (xC ,h-yMargin-y1Margin), axisColor, 1)
    for yC,yV in zip(yCoor, yValues):
        cv2.putText(base,str(round(yV,2)),(5, yC + 4), font, 0.4, axisColor,1,cv2.LINE_AA)
        cv2.line(base, (xMargin-4 , yC), (xMargin , yC), axisColor, 1)

    dataX = (dataX - xMin)*scaleX + xMargin
    dataY = h - yMargin - y1Margin - (dataY - yMin)*scaleY

    zeroValueY = int(h - yMargin - y1Margin + yMin*scaleY)

    if type == "line":
        pts = np.vstack((dataX,dataY)).astype(np.int32).T
        cv2.polylines(base, [pts], isClosed=False, color=color)
    elif type == "rectangle":
        for i, yVal in enumerate(dataY):
            incDataX = int(dataX[1] - dataX[0])
            xVal = int(dataX[i])
            tl = ( xVal,int(yVal))
            br = ( xVal + incDataX, zeroValueY)
            cv2.rectangle(base, tl, br, color, 1, cv2.LINE_AA, 0)
    else:
        pts = np.vstack((dataX,dataY)).astype(np.int32).T
        for i in pts:
            cv2.circle(base,(i[0],i[1]),3,color, 1, cv2.LINE_AA, 0)

    cv2.line(base, (xMargin,h-yMargin-y1Margin), (w-xMargin,h-yMargin-y1Margin), axisColor, 1)
    cv2.line(base, (xMargin,h-yMargin-y1Margin), (xMargin, yMargin+y1Margin), axisColor, 1)
    cv2.putText(base,title,(xMargin,yMargin), font, 0.5,axisColor,1,cv2.LINE_AA)


    return base

def plotHistogram(dataH, bins = 'auto', roundFactor = 1, figure = [], axisXRange = [], axisYRange = [], color = (0,255,255), title = "Histogram"):

    minData = np.amin(dataH)
    maxData = np.amax(dataH)

    hist, bin_edges = histArea = np.histogram(dataH, bins)

    if axisXRange==[]:
        axisXRange=[minData, maxData]

    if axisYRange==[]:
        axisYRange=[0, np.amax(histArea[0])]

    figure = plotData(bin_edges, hist, axisXRange, axisYRange, color, figure, [], "rectangle", title)
    return figure

def drawFeatures(img, contoursPoints, contoursFiltered, boundinBox, data):

    pt1 = (boundinBox[0], boundinBox[1])
    pt2 = (boundinBox[2], boundinBox[3])

    cv2.drawContours(img, [contoursPoints], -1, (128,255,255), 1, cv2.LINE_AA)
    cv2.drawContours(img, [contoursFiltered], -1, (255,128,255), 1, cv2.LINE_AA)
    cv2.rectangle(img, pt1, pt2, (255,255,128), 1, cv2.LINE_AA, 0)
    cv2.circle(img,(data[2],data[3]),10,(255,128,128), 1, cv2.LINE_AA, 0)
    cv2.circle(img,(data[9],data[10]),10,(128,255,128), 1, cv2.LINE_AA, 0)
    cv2.ellipse(img, (data[16],data[17]), (data[14],data[15]), data[6], startAngle=0, endAngle=360, color=(128, 128, 255), thickness = 1)
    cv2.arrowedLine(img, (data[2],data[3]), (data[9],data[10]), color=(200, 200, 200), thickness=1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,str(data[4])[:7],(5,470), font, 0.4,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,str(data[5])[:7],(65,470), font, 0.4,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,str(data[6])[:5],(125,470), font, 0.4,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,str(data[7])[:5],(175,470), font, 0.4,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,str(data[8])[:5],(225,470), font, 0.4,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,str(data[11])[:5],(275,470), font, 0.4,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,str(data[12])[:5],(325,470), font, 0.4,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,str(data[13])[:5],(375,470), font, 0.4,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,str(data[14])[:5],(430,470), font, 0.4,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,str(data[15])[:5],(480,470), font, 0.4,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,str(data[18])[:5],(530,470), font, 0.4,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,str(data[19])[:5],(580,470), font, 0.4,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,str(data[1])[:5],(625,470), font, 0.4,(255,255,255),1,cv2.LINE_AA)
    legend = " area   perimeter angle  aRatio solidity meanB meanG meanR minAxis majAxis angleIlu distIlu ID"
    cv2.putText(img,legend,(10,455), font, 0.4,(255,255,255),1,cv2.LINE_AA)

    return img

def drawFigures(allData, color, figPos, figAngle, figArea, figPerimeter, figAspectRatio, figSolidity, figBmean, figGmean, figRmean, figAIlu, figDIlu, figDAIlu):

    # POSITION
    figPos = fn.plotData(allData[:,0], allData[:,1], [0,w], [0,h], color, figurePos, [4,4], "circle", "Position " + legend)
    # ANGLE
    figAngle = fn.plotHistogram(allData[:,4], title = "Histogram Angle", figure = figureAngle, color = color, bins=10, axisXRange =[0, 180], axisYRange =[0, 100])
    # AREA
    figArea = fn.plotHistogram(allData[:,2], roundFactor = 1000, title = "Histogram Area", figure = figureArea, axisXRange =[8000, 80000], axisYRange =[0, 45], color = color)
    # PERIMETER
    figPerimeter = fn.plotHistogram(allData[:,3], roundFactor = 100, title = "Histogram Perimeter", figure = figurePerimeter, axisXRange =[500, 2500], axisYRange =[0, 45], color = color)
    # ASPECT RATIO,
    figAspectRatio = fn.plotHistogram(allData[:,5], title = "Histogram AspectRatio", figure = figureAspectRatio, axisXRange =[0.2, 1], color = color, axisYRange =[0, 45])
    # SOLIDITY,
    figSolidity = fn.plotHistogram(allData[:,6], title = "Histogram Solidity", figure = figureSolidity, axisXRange =[0.5, 1], color = color, axisYRange =[0, 45])
    # mean B values (color)
    figBmean = fn.plotHistogram(allData[:,9], title = "Histogram B mean Values", figure = figureBmean, axisXRange =[0, 128], axisYRange =[0, 45], color = color)
    # mean G values (color)
    figGmean = fn.plotHistogram(allData[:,10], title = "Histogram G mean Values", figure = figureGmean, axisXRange =[0, 128], axisYRange =[0, 45], color = color)
    # mean R values (color)
    figRmean = fn.plotHistogram(allData[:,11], title = "Histogram R mean Values", figure = figureRmean, axisXRange =[0, 128], axisYRange =[0, 45], color = color)
    # ANGLE ILUMINATION
    figAIlu = fn.plotHistogram(allData[:,16], bins = 10, title = "Histogram Angle Ilumination", figure = figureAIlu, color = color, axisXRange = [0, 360], axisYRange =[0, 150])
    # DISTANCE ILUMINATION
    figDIlu = fn.plotHistogram(allData[:,17], title = "Histogram Distance Ilumination", figure = figureDIlu, axisXRange =[0, 80], axisYRange =[0, 45], color = color)
    # DISTANCE/ANGLE ILUMINATION
    figDAIlu = fn.plotData(allData[:,16], allData[:,17], [0, 360], [0, 80], color, figureDAIlu, [4,4], "circle", "Angle / Distance Ilumination")

    return figPos, figAngle, figArea, figPerimeter, figAspectRatio, figSolidity, figBmean, figGmean, figRmean, figAIlu, figDIlu, figDAIlu
