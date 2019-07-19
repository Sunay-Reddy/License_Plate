import cv2
import numpy as np
from imutils.perspective import four_point_transform
from skimage.filters import threshold_local
from skimage import measure
import imutils


image = cv2.imread('C:/Users/SUNAY REDDY/PycharmProjects/license_plate/test1.jpg')
resized = imutils.resize(image, width = 1000)
ratio = image.shape[0]/ float(resized.shape[0])
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)
cv2.imshow("canny", edged)
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
count=0
for c in cnts:
    #cv2.drawContours(resized, [c], -1, (0, 0, 255), -1)
    (x, y, w, h) = cv2.boundingRect(c)
    """
    rectangle_peri = 2*(w + h)
    contour_peri = cv2.arcLength(c, True)
    #print(contour_area/rectangle_area)
    if float(contour_peri/rectangle_peri) >= 0.95 :
        count+=1
        cv2.drawContours(resized, [c], -1, (0, 0, 255), -1)

    #cv2.rectangle(resized, (x, y), (x+w, y+h), (0, 0, 255), 2)
    """
    ar = w/ float(h)
    M = cv2.moments(c)
    area = M["m00"]
    if ar>=2 and ar<=5 and (w*h)>=1200:
        #M = cv2.moments(c)
        #cX = int((M["m10"]/M["m00"]) * ratio)
        #cY = int((M["m01"]/M["m00"]) * ratio)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)

        if len(approx) == 4 and area>=1500:
            screen = approx
            count+=1
            #cv2.drawContours(resized, [c], -1, (0, 0, 255), -1)
            #cv2.drawContours(resized, [c], -1, (0, 0, 255), -1)
            lic_plate = four_point_transform(resized, screen.reshape(4,2))
            lic_plate = imutils.resize(lic_plate, width = 200)
            #adaptive thresholding
            V = cv2.split(cv2.cvtColor(lic_plate, cv2.COLOR_BGR2HSV))[2]
            T = threshold_local(V, 29, offset=15, method='gaussian')
            thresh = (V>T).astype("uint8") * 255
            thresh = cv2.bitwise_not(thresh)
            thresh = imutils.resize(thresh, width = 200)

            labels = measure.label(thresh, neighbors=8, background=None)
            chars = np.zeros(thresh.shape, dtype = "uint8")
            for label in np.unique(labels):
                if label == 0:
                    continue
                labelmask = np.zeros(thresh.shape, dtype = "uint8")
                labelmask[labels == label] = 255
                cnts = cv2.findContours(labelmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                #cnts = cnts[0] if imutils.is_cv2() else cnts[1]
                cnts = imutils.grab_contours(cnts)
                #license_plates = lic_plate
                for c in  cnts:
                    M = cv2.moments(c)
                    area = M["m00"];
                    #print(area)
                    if area>=100:
                        (x, y, w, h) = cv2.boundingRect(c)
                        cv2.rectangle(lic_plate, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        #cv2.drawContours(lic_plate, [c], -1, (0, 0, 255), -1)
                #cv2.imshow('loop', labelmask)
                #cv2.waitKey(0)

            cv2.imshow('thresh', thresh)
            cv2.imshow('lic_plate', lic_plate)


#print(count)
cv2.imshow("resized", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()