import io
import libconf
import argparse
import cv2
import numpy as np
import cFunctions as fn
from os import listdir, getcwd
from os.path import isfile, join
from matplotlib import pyplot as plt
import random
import pandas as pd

# TO USE THIS -> python cAnalysis.py cFile.cfg

parser = argparse.ArgumentParser()
parser.add_argument("configFile")
args = parser.parse_args()
cwd = getcwd()

print("Reading configuration file: " + cwd + "\\" + args.configFile)

with io.open(args.configFile, encoding='utf-8') as f:
    config = libconf.load(f)

print ("IDs:")
print (config.IDs)

cwd = config.basePath

# Read background image for threshold
imgBk = cv2.imread(join(config.basePath, config.backgroundFile), 1)

# Read background images for combination
backgroundFolder = join(config.basePath, config.backgroundPath)
backgroundFiles = listdir(backgroundFolder)
nBackFiles = len(backgroundFiles)
print("Backgroud files: " + str(nBackFiles))

# Convert BGR to HSV
imgBkHSV = cv2.cvtColor(imgBk, cv2.COLOR_BGR2HSV) 

# Detection of background value for threshold
imgBkHSV_S = cv2.split(imgBkHSV)
meanValueBK = cv2.mean(np.array(imgBkHSV_S[config.channel]))
meanValueBK = meanValueBK[0]
print("Mean Value: " + str(meanValueBK))
print("Range: " + str(meanValueBK-config.threshold) + " - " + str(meanValueBK+config.threshold))

allData = []
accepted_extensions = ["jpg", "png"]
#for item in config.items:
for i in range(len(config.IDs)):
    item = config.IDs[i]
    currentFolder = config.basePath + item + "/"
    print('Reading folder: ' + currentFolder)
    imageNumber = 0

    fileNameList = [fn for fn in listdir(currentFolder) if fn.split(".")[-1] in accepted_extensions]

    for fileName in fileNameList:
        imageName = currentFolder + fileName
        img = cv2.imread(imageName, 1)
        imgY = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        # Object detection with threshold
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        imgHSV_S = cv2.split(imgHSV)

        mask = cv2.inRange(imgHSV_S[config.channel], meanValueBK-config.threshold, meanValueBK+config.threshold)
        mask = cv2.bitwise_not(mask)

        object = mask.copy()
        kernel = np.ones((5, 5), np.uint8)
        # object = cv2.dilate(object, kernel, iterations = 1)
        # object = cv2.dilate(object, kernel, iterations = 1)
        object = cv2.erode(object, kernel, iterations = 1)


        # Detect contours of all objects in the image
        object, contours, hierarchy = cv2.findContours(object,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        # Detect the bigger object and obtains its bounding box
        x, y, w, h, index = fn.boundingBox(contours)
        # Create a mask of the bigger object to remove noise. The original one has noise because of the light
        maskObject = np.zeros((object.shape), np.uint8)
        cv2.drawContours(maskObject, [contours[index]], -1, (255,255,255), -1, cv2.LINE_AA)
        # Defines the set of points of the contour. The original ones and the filtered ones
        # contoursPoints, contoursFiltered = fn.defineContours(contours, config.aproximation, config.numberMinPointsThreshold, config.numberMaxPointsThreshold)
        contoursPoints = cv2.approxPolyDP(contours[index], config.aproximation, 1)
        contoursFiltered = contoursPoints

        # Save results in an XML file to use it with TensorFlow
        height, width, channels = img.shape
        fileNameSave_ = config.resultImageFileName + "_" + item + "_" + str(imageNumber) + ".jpg"
        fn.saveFileXML(fileNameSave_, config.resultFolder, config.basePath, height, width, channels, item, x, y, w, h);

        # Saves the new image. The new image is a combination of the original one and one background.
        # Combination with background. If backgrounds folder is empty, combination is not done
        if nBackFiles > 0:
            imgBkFileName = join(backgroundFolder, backgroundFiles[random.randint(0, nBackFiles - 1)])
            imgBakground = cv2.imread(imgBkFileName, 1)
             imgComb = fn.combineImages(img, imgBakground, object)
        else:
            imgComb = img

        # save the image jpg
        fileNameSave = join(config.basePath + config.resultFolder, fileNameSave_)
        cv2.imwrite(fileNameSave, imgComb)

        # save the mask
        fileNameSave = fileNameSave[:fileNameSave.find(".")] + ".png"
        cv2.imwrite(fileNameSave, maskObject)

        print (imageName + " --> " + fileNameSave)

        # Extract features of the object
        data = fn.features(img, maskObject, contours[index])
        imgInfo = [fileNameSave_, item]
        imgInfo.extend(data)
        allData.append(imgInfo)

        imageNumber = imageNumber + 1

        # Show results
        boundinBox = [x, y, x+w, y+h]
        imgFeatures = fn.drawFeatures(imgComb, contoursPoints, contoursFiltered, boundinBox, imgInfo)
        cv2.putText(mask,"MASK",(25,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(object,"OBJECT",(25,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(maskObject,"MASK_OBJECT",(25,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
        mask = cv2.resize(mask,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        object = cv2.resize(object,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        maskObject = cv2.resize(maskObject,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        numpy_vertical_1 = np.vstack((mask, object))
        numpy_vertical_1 = np.vstack((numpy_vertical_1, maskObject))

        imgR = cv2.resize(imgHSV,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        imgR_S = cv2.split(imgR)
        numpy_vertical_2 = np.vstack((imgR_S[0], imgR_S[1]))
        numpy_vertical_2 = np.vstack((numpy_vertical_2, imgR_S[2]))

        cv2.imshow('Masks', numpy_vertical_1)
        cv2.imshow('HSV', numpy_vertical_2)
        cv2.imshow('img', imgComb)

        k = cv2.waitKey(config.waitTime)
        if k==27:
            break
    if k==27:
        break

column_name = ['filename', 'class', 'cx', 'cy', 'area', 'perimeter', 'angle', 'aspectRatio', 'solidity', 'cy_iluWeight', 'cx_iluWeight', 'mean_Blue', 'mean_Green', 'mean_Red', 'minorAxis', 'majorAxis', 'xEllipse', 'yEllipse', 'angleIllumination', 'distanceIllumination']
allData_df = pd.DataFrame(allData, columns=column_name)
allData_df.to_csv(config.basePath + config.resultFolder + "/" + config.dataFile, index=None)
print("csv file saved to " + config.basePath + config.resultFolder + "/" + config.dataFile)


cv2.destroyAllWindows()
