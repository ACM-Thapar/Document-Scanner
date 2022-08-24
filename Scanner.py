import cv2
import numpy as np
import os
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

imgQ = cv2.imread('Forms\\test2.jpg')

#imgQ = cv2.resize(imgQ,(w//3,h//3))
per = 25
roi = [[50,60,300,50],'box','Sign']


orb = cv2.ORB_create(1000)
kp1,des1 = orb.detectAndCompute(imgQ,None)
#imgKp1 = cv2.drawKeypoints(imgQ,kp1,None)

path = 'Forms'
myPicList = os.listdir(path)
print(myPicList)
for j,y in enumerate(myPicList):
    img = cv2.imread(path + "/" + y)
    #cv2.imshow(y,img)
    kp2,des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2,des1)
    matches = list(matches)
    matches.sort(key= lambda x: x.distance)
    good = matches[:int(len(matches)*(per/100))]
    imgMatch = cv2.drawMatches(img,kp2,imgQ,kp1,good[:100],None,flags=2)
    cv2.imshow(y,imgMatch)

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dstPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)

    M,_ = cv2.findHomography(srcPoints,dstPoints,cv2.RANSAC,5.0)
    h, w, c = img.shape
    imgScan = cv2.warpPerspective(img,M,(w,h))
    cv2.imshow(y, imgScan)

#cv2.imshow("KeyPointsQuery",imgKp1)
#cv2.imshow("Output",imgQ)
cv2.waitKey(0)

