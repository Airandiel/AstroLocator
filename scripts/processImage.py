from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2


class StarImageProcess():
    def find_stars(self, draw = False): # returns list of brightest stars [[locx, locy], brightness]
        image, height, width = self.read_picture("../pictures/IMG_9548.jpg")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cnts = self.make_proper_contours(gray)
        starList = self.get_list_of_stars_with_magnitude(cnts, image, True)
        if draw:
            print(starList)
            cv2.imshow("Image", image)
            cv2.waitKey(0)
        return(starList)

    def get_list_of_stars_with_magnitude(self, cnts, image = None, draw = False):
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        starList = []
        for c in cnts:
            # compute the center of the contour
            area = cv2.contourArea(c)
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            starList.append([[cX, cY], area])
            if draw:
                cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
                cv2.circle(image, (cX, cY), int(np.sqrt(area)), (255, 255, 255), -1)
                cv2.putText(image, "center", (cX - 20, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return starList


    def read_picture(self, path = '../pictures/stars.jpg'):
        image = cv2.imread(path)
        h, w = image.shape[:2]
        return image, h, w

    def make_proper_contours(self, image, minNumberOfStars = 15, maxNumberOfStars = 500):
        numberOfContours = 0
        thresholdBrightness = 180
        while numberOfContours < minNumberOfStars or numberOfContours > maxNumberOfStars:
            thresh = cv2.threshold(image, thresholdBrightness, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.erode(thresh, None, iterations=1)
            thresh = cv2.dilate(thresh, None, iterations=1)  # make the dots bigger
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]
            numberOfContours = len(cnts)
            if numberOfContours < minNumberOfStars:
                thresholdBrightness -= 5
            elif numberOfContours > maxNumberOfStars:
                thresholdBrightness += 5
                
        return cnts


# print(StarImageProcess().find_stars(True))


# labels, numLabels = measure.label(thresh, neighbors=8, connectivity=1, return_num=True)
#
# while( numLabels > 500):
#     thresh = cv2.erode(thresh, None, iterations=1)
#     # thresh = cv2.dilate(thresh, None, iterations=1) # make the dots bigger
#     labels, numLabels = measure.label(thresh, neighbors=8, connectivity=1, return_num=True)
#
# mask = np.zeros(thresh.shape, dtype="uint8")
# stars = []
# print(numLabels)
#
# # loop over the unique components
# # counter = 0
# for label in np.unique(labels):
#     # if this is the background label, ignore it
#     # print(counter)
#     # counter += 1
#     if label == 0:
#         continue
#     # otherwise, construct the label mask and count the
#     # number of pixels
#     labelMask = np.zeros(thresh.shape, dtype="uint8")
#     labelMask[labels == label] = 255
#     numPixels = cv2.countNonZero(labelMask)
#
#     # if the number of pixels in the component is sufficiently
#     # large, then add it to our mask of "large blobs"
#     stars.append()
#     if numPixels > 0:
#         mask = cv2.add(mask, labelMask)
#
# print(numPixels)
# cv2.imshow('image',mask)
# cv2.waitKey(0)
