import cv2
import numpy as np
from processData import ProcessData


def printStars(stars, id1, id2, id3):
    width = 600
    height = 400
    image = np.zeros((height, width, 3), np.uint8)

    miny = min(stars[:, 2])
    maxy = max(stars[:, 2])
    print(miny, maxy)
    stars[:, 1] *= np.cos(stars[:, 2])
    minx = min(stars[:, 1])
    maxx = max(stars[:, 1])
    maxBright = max(stars[:, 3])
    oldWidth = maxx - minx
    oldHeight = maxy - miny
    for star in stars:
        starx = (star[1] - minx) * width / oldWidth
        stary = height - (star[2] - miny) * height / oldHeight
        #starx = (star[1])*width/(2*np.pi)
        #stary = height - (star[2]+np.pi/2)*height/np.pi
        if starx >= width:
            starx = width - 1
        elif starx < 0:
            starx = 0

        if stary >= height:
            stary = height - 1
        elif stary < 0:
            stary = 0

        cv2.circle(image, (int(starx), int(stary)), int(star[3]*3/maxBright), (200, 200, 200), 1)
        if star[0] == id1 or star[0] == id2 or star[0] == id3:
            cv2.circle(image, (int(starx), int(stary)), 5, (0, 200, 0), 2)

        # if counter == int(len(stars) / 2):
        #     cv2.circle(image, (int(stary), int(starx)), 5, (0, 200, 0), 2)
        # counter += 1
    cv2.imshow("Image", image)
    cv2.waitKey(0)


def initStars(id, id2, id3):
    p = ProcessData()
    stars = p.export_from_nothing(id)
    printStars(stars, id, id2, id3)


initStars(75035, 42990, 57459)
