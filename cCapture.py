import io
import libconf
import argparse
import cv2
import numpy as np
import cFunctions as fn
from os import listdir, getcwd
from os.path import isfile, join

imageNumber = 0
basePath = "";
cam0 = cv2.VideoCapture(0)

save = 0

while 1:
    ret_val, img = cam0.read()
    cv2.imshow('frame', img)

    k = cv2.waitKey(200)

    fileNameSave = "img" + str(imageNumber) + ".jpg"
    fileNameSave = basePath + fileNameSave
    print ("Saving --> " + fileNameSave)
    cv2.imwrite(fileNameSave, img)
    imageNumber = imageNumber + 1

    if k==27:
        break

cv2.destroyAllWindows()
