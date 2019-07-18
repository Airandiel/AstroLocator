from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import os.path, time


class Worker(object):
    def __init__(self, img):
        self.img = img
        self.get_exif_data()
        self.date = self.get_date_time()
        super(Worker, self).__init__()

    def get_exif_data(self):
        exif_data = {}
        info = self.img._getexif()
        if info:
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                if decoded == "GPSInfo":
                    gps_data = {}
                    for t in value:
                        sub_decoded = GPSTAGS.get(t, t)
                        gps_data[sub_decoded] = value[t]

                    exif_data[decoded] = gps_data
                else:
                    exif_data[decoded] = value
        self.exif_data = exif_data
        # return exif_data

    def get_date_time(self):
        if 'DateTime' in self.exif_data:
            date_and_time = self.exif_data['DateTime']
            return date_and_time


class StarImageProcess():
    @staticmethod
    def get_date_taken(path):
        img = Image.open(path)
        image = Worker(img)
        return image.date

    def find_stars(self, draw=False, path="../pictures/IMG_9548.jpg",
                   dateTime=""):  # returns list of brightest stars [[locx, locy], brightness]
        image, height, width = self.read_picture(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cnts = self.make_proper_contours(gray)
        starList = self.get_list_of_stars_with_magnitude(cnts, image, True)
        if draw:
            print(starList)
            cv2.imshow("Image", image)
            cv2.waitKey(0)
        # date = StarImageProcess.get_date_taken(path)
        # if dateTime == "":
        #     date = StarImageProcess.get_date_taken(path)

        return starList

    def get_list_of_stars_with_magnitude(self, cnts, image=None, draw=False):
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        starList = []
        h, w = image.shape[:2]
        for c in cnts:
            # compute the center of the contour
            area = cv2.contourArea(c)
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            flag = False
            for star in starList:
                if ((star[0][0] - w / 300 < cX < star[0][0] + w / 300) or (
                        star[0][1] - h / 300 < cY < star[0][1] + h / 300)):
                    flag = True
                    continue
            if not flag:
                starList.append([[cX, cY], area])
                if draw:
                    cv2.drawContours(image, [c], -1, (0, 255, 0), 5)
                    # cv2.circle(image, (cX, cY), int(np.sqrt(area)), (255, 255, 255), -1)
                    # cv2.putText(image, "s", (cX - 20, cY - 20),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.imwrite("contours.jpg", image)

        return starList

    def read_picture(self, path='../pictures/stars.jpg'):
        image = cv2.imread(path)
        h, w = image.shape[:2]
        return image, h, w

    def make_proper_contours(self, image, minNumberOfStars=25, maxNumberOfStars=500):
        numberOfContours = 0
        thresholdBrightness = 180
        image = cv2.Sobel(image,cv2.CV_8UC1,1,0,ksize=5)
        cv2.imwrite("image.jpg", image)
        while numberOfContours < minNumberOfStars or numberOfContours > maxNumberOfStars:
            thresh = cv2.threshold(image, thresholdBrightness, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.erode(thresh, None, iterations=2)
            thresh = cv2.dilate(thresh, None, iterations=5)
            cv2.imwrite("treshold.jpg", thresh)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]
            numberOfContours = len(cnts)
            if numberOfContours < minNumberOfStars:
                thresholdBrightness -= 5
            elif numberOfContours > maxNumberOfStars:
                thresholdBrightness += 5

        return cnts


localizations, date = StarImageProcess().find_stars(draw=True, path="../pictures/IMG_9527.jpg")
# print(date)

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
