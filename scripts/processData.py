import csv
import numpy as np
import itertools
from random import shuffle
import copy
from processImage import StarImageProcess
from scipy.spatial import ConvexHull


def find_most_similar(starsArray, databasePath='../database/hygdata_v3.csv'):
    p = ProcessData()
    database = p.read_and_process_database(databasePath)
    p.search_most_similar_stars(starsArray, database)


class Star():
    def __init__(self, x=0, y=0, bright=0, cell=[]):
        if len(cell) == 4:
            self.id = cell[0]
            self.x = cell[1] * np.cos(cell[2])
            self.y = cell[2]
            self.bright = cell[3]
        elif len(cell) == 2:
            self.x = cell[0][0]
            self.y = cell[0][1]
            self.bright = cell[1]
        else:
            self.x = x
            self.y = y
            self.bright = bright

    def dist(self, other):
        return pow(self.x - other.x, 2) + pow(self.y - other.y, 2)

    @staticmethod
    def sort_dist(first, second):
        return pow(first.x - second.x, 2) + pow(first.y - second.y, 2)

    @staticmethod
    def dist_ratio(first, second, third):
        # middle to longest, shortest to longest
        fs = first.dist(second)
        st = second.dist(third)
        ft = first.dist(third)
        arr = sorted([fs, ft, st])
        return arr[1] / arr[2], arr[0] / arr[2]

    @staticmethod
    def sort_three_by_distances(first, second, third):
        fs = first.dist(second)
        st = second.dist(third)
        ft = first.dist(third)
        if fs > st and fs > ft:
            return first, second
        elif st > fs and st > ft:
            return second, third
        else:
            return first, third

    @staticmethod
    def angles_convex_hull(list):
        # return angles of hull
        points = np.array([list.x, list.y])
        hull = ConvexHull(points)
        end = len(hull)
        one = np.concatenate((hull.vertices[end - 1:end], hull.vertices[0:end - 1]))
        two = np.concatenate((hull.vertices[1:end], hull.vertices[0:1]))
        diff = points[one] - points[hull.vertices]
        one = np.hypot(diff[:, 0], diff[:, 1])
        maxValue, maxIndex = max(one)
        diff = points[two] - points[hull.vertices]
        two = np.hypot(diff[:, 0], diff[:, 1])
        angles = np.arctan(one/two)
        if maxIndex != 0:
            angles = np.concatenate((angles[maxIndex:len(angles)], angles[0:maxIndex]))
        return angles




class ProcessData:
    def search_most_similar_stars(self, starsArray, database, databasePath='../database/hygdata_v3.csv'):
        ### Firstly take 3 brightest stars from image - change it in future !!!!
        first = Star(cell=starsArray[0])
        second = Star(cell=starsArray[1])
        third = Star(cell=starsArray[2])

        # calculate distance relations because angles would take to much time
        # ml - middle to longest, sl - shortest to longest

        ml, sl = Star.dist_ratio(first, second, third)
        first, second = Star.sort_by_distances(first, second, third)
        ### READ DATABASE
        file = open(databasePath, 'r')
        databaseRaw = self.read_remove_unimportant_columns_and_convert(file)
        databaseRawPass = copy.copy(databaseRaw)
        databaseRawPass[:, 3] = pow(10, (-14.18 - databaseRawPass[:, 3]) / 2.5)
        for mag in range(10, 65, 5):
            database = self.reduce_too_dark_and_convert_magnitude(copy.copy(databaseRaw), magThreshold=float(mag / 10))
            if len(database) > 0:
                # fc - first choosen, sc - second choosen, tc - third choosen
                # make for 50 nearests stars
                n = 100
                if len(database) < n:
                    n = len(database)
                children = [i for i in range(len(database))]
                shuffle(children)
                stats = []
                counter = 0
                for child in children:
                    if counter % np.ceil(len(database) / 10) == 0:
                        print(int(counter / np.ceil(len(database) / 100)), "%")
                    fc = Star(cell=database[child])
                    for setOfStars in itertools.combinations(range(n), 2):
                        if setOfStars[0] == int(n / 2) or setOfStars[1] == int(n / 2):
                            continue
                        else:
                            mod1 = setOfStars[0]
                            mod2 = setOfStars[1]
                            if child - int(n / 2) + setOfStars[0] < 0:
                                mod1 = len(database) - setOfStars[0] + int(n / 2) - child - 1
                            elif child - int(n / 2) + setOfStars[0] >= len(database):
                                mod1 -= len(database)
                            if child - int(n / 2) + setOfStars[1] < 0:
                                mod2 = len(database) - setOfStars[1] + int(n / 2) - child - 1
                            elif child - int(n / 2) + setOfStars[1] >= len(database):
                                mod2 -= len(database)

                            sc = Star(cell=database[child - int(n / 2) + mod1])
                            tc = Star(cell=database[child - int(n / 2) + mod2])
                            # cml - choosen middle to longest, csl - choosen shortest to longest
                            cml, csl = Star.dist_ratio(fc, sc, tc)
                            longests = min(cml, ml) / max(cml, ml)
                            shortests = min(csl, sl) / max(csl, sl)
                            stat = longests * shortests
                            if stat > 0.99:
                                arr = sorted([fc, sc, tc], key=lambda x: x.bright)
                                ratiofs1 = first.bright / second.bright
                                ratiofs2 = arr[2].bright / arr[1].bright
                                ratiost1 = second.bright / third.bright
                                ratiost2 = arr[1].bright / arr[0].bright
                                ratioft1 = first.bright / third.bright
                                ratioft2 = arr[2].bright / arr[0].bright
                                ratiofs = min(ratiofs1, ratiofs2) / max(ratiofs1, ratiofs2)
                                ratiost = min(ratiost1, ratiost2) / max(ratiost1, ratiost2)
                                ratioft = min(ratioft1, ratioft2) / max(ratioft1, ratioft2)
                                if ratiofs * ratiost * ratioft > 0.90:
                                    fsc, ssc = Star.sort_by_distances(fc, sc, tc)
                                    starList = [fc.id, sc.id, tc.id]
                                    if self.find_next(first, second, Star(cell=starsArray[3]), starsArray, 3, fsc, ssc,
                                                      databaseRawPass, starList):
                                        stats.append(starList)
                                        print(starList)
                                    # stats.append([fc.id, sc.id, tc.id])
                                    # print([fc.id, sc.id, tc.id])
                    counter += 1
                # print(stats)
        return stats

    # returns True if at least five stars are correct, False, other way
    def find_next(self, polygon1, starArray, noOfChecked, polygon2, databaseRaw, starsList, n=5000):
        if noOfChecked == 6:
            return True

        originalAngles = Star.angles_convex_hull(polygon1)
        # polygon1.sort(cmp=Star.sort_dist)
        # polygon2.sort(cmp=Star.sort_dist)

        # fs - first star, ss - second star ...
        # fsc - first star choosen, ssc - second star choosen ...
        # ml, sl = Star.dist_ratio(polygon1[0], polygon1[1], polygon1[2])
        index = np.where(databaseRaw[:, 0] == polygon2[0].id)
        index = index[0][0]
        if index - n < 0:
            database = np.concatenate(
                (databaseRaw[len(databaseRaw) + index - n - 1:len(databaseRaw) - 1], databaseRaw[0:index + n]))
        elif index + n > len(databaseRaw):
            database = np.concatenate(
                (databaseRaw[index - n:len(databaseRaw) - 1], databaseRaw[0:index + n - len(databaseRaw)]))
        else:
            database = databaseRaw[index - n:index + n]
        for child in database:
            tsc = Star(cell=child)
            #cml, csl = Star.dist_ratio(fsc, ssc, tsc)
            #longests = min(cml, ml) / max(cml, ml)
            #shortests = min(csl, sl) / max(csl, sl)

            stat = self.calculate_similarity_of_polygons(originalAngles, np.concatenate((polygon2, [tsc])))
            if stat < 0.1:
                #fsc, ssc = Star.sort_by_distances(fsc, ssc, tsc)
                #result = self.find_next(fs, ss, Star(cell=starArray[noOfChecked + 1]), starArray, noOfChecked + 1, fsc,
                #                        ssc,
                #                        databaseRaw,
                #                        starsList)
                newAngles = np.concatenate((originalAngles, [Star(cell=starArray[noOfChecked+1])]))
                newPolygon = np.concatenate((polygon2, [tsc]))
                result = self.find_next(newAngles, starArray, noOfChecked + 1, newPolygon,
                                       databaseRaw,
                                       starsList)
                if result:
                    starsList.append(tsc.id)
                    return True
        return False

    def calculate_similarity_of_polygons(self, anglesPolygon1, polygon2):
        # compare angles in convex hull
        angles = Star.angles_convex_hull(polygon2)
        if len(anglesPolygon1) != len(angles):
            return 0
        else:
            return sum(angles - anglesPolygon1)



    def read_and_process_database(self, databasePath):
        file = open(databasePath, 'r')
        database = self.read_remove_unimportant_columns_and_convert(file)
        database = self.reduce_too_dark_and_convert_magnitude(database)
        return database

    def read_remove_unimportant_columns_and_convert(self, file):
        reader = csv.reader(file, delimiter=',')
        database = np.array(list(reader))[1:, [0, 7, 8, 13]]
        #### id, ra, dec, mag
        database = database.astype(float)
        hours = np.floor(database[:, 1])
        minutes = np.floor((database[:, 1] - hours) * 60)
        seconds = ((database[:, 1] - hours) * 60 - minutes) * 60
        database[:, 1] = 2 * np.pi * ((hours / 24) + minutes / 24 / 60 + seconds / 24 / 60 / 60)
        database[:, 2] = database[:, 2] * np.pi / 180
        # database[:, 1] = np.floor(database[:,1])/12*np.pi
        return database

    def reduce_too_dark_and_convert_magnitude(self, database, magThreshold=6.5):
        database = database[(database[:, 3] < magThreshold)]
        database[:, 3] = pow(10, (-14.18 - database[:, 3]) / 2.5)
        database = database.tolist()
        database.sort(key=self.stars_distance_sort)  # Optional sorting accoding to distance to center
        return database

    def combinations(self, iterable, r):
        # combinations(range(4), 3) --> 012 013 023 123
        n = len(iterable)
        if r > n:
            return
        indices = list(range(r))
        yield tuple(iterable[i] for i in indices)
        while True:
            for i in reversed(range(r)):
                if indices[i] != i + n - r:
                    break
            else:
                return
            indices[i] += 1
            for j in range(i + 1, r):
                indices[j] = indices[j - 1] + 1
            yield tuple(iterable[i] for i in indices)

    def stars_distance_sort(self, star):
        return pow(star[1] * np.cos(star[2]), 2) + pow(star[2], 2)

    def export_stars_to_print(self, databaseRaw, id, noOfStars=3000):
        database = self.reduce_too_dark_and_convert_magnitude(databaseRaw, 6.5)
        database = np.array(database)
        index = np.where(database[:, 0] == id)
        index = index[0][0]
        if (index - noOfStars < 0):
            output = database[len(database + index - noOfStars) - 1:len(database) - 1] + database[0:index + noOfStars]
        elif index + noOfStars > len(database):
            output = database[index - noOfStars:len(database) - 1] + database[0:index + noOfStars - len(database)]
        else:
            output = database[index - noOfStars:index + noOfStars]
        return output

    def export_from_nothing(self, id):
        databasePath = '../database/hygdata_v3.csv'
        file = open(databasePath, 'r')
        database = self.read_remove_unimportant_columns_and_convert(file)
        # print(max(database[:, 1]))
        return self.export_stars_to_print(database, id)


find_most_similar(StarImageProcess().find_stars())
# find_most_similar([[[774, 267], 81.5], [[827, 54], 26.0], [[398, 404], 25.5], [[388, 240], 22.5], [[771, 157], 16.5],
#                   [[770, 362], 14.0], [[226, 184], 14.0], [[689, 446], 13.5], [[875, 451], 12.0], [[206, 367], 12.0],
#                   [[433, 170], 12.0], [[384, 314], 11.0], [[749, 160], 10.0], [[298, 310], 8.5], [[839, 355], 6.0],
#                   [[692, 336], 6.0], [[429, 335], 6.0], [[377, 288], 6.0], [[37, 560], 4.0], [[457, 557], 4.0],
#                   [[843, 449], 4.0], [[597, 446], 4.0], [[679, 422], 4.0], [[636, 409], 4.0], [[459, 338], 4.0],
#                   [[328, 336], 4.0], [[730, 330], 4.0], [[325, 309], 4.0], [[712, 302], 4.0], [[416, 288], 4.0],
#                   [[46, 241], 4.0], [[519, 198], 4.0], [[330, 47], 4.0], [[717, 13], 4.0], [[281, 13], 4.0]]
#                  )
# 165 [21228.0, 79640.0, 3413.0]
# 386 [8087.0, 108528.0, 9655.0]
# 216 [102895.0, 43700.0, 51348.0]
# 1969 [68954.0, 87844.0, 31518.0]
# 1746 [113262.0, 25858.0, 14877.0]
# 683 [39073.0, 89400.0, 83991.0]

# 75035.0, 42990.0, 57459.0
# 56174.0, 44343.0, 71125.0
# 55539.0, 70274.0, 85410.0
