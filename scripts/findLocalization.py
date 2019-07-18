import numpy as np
from DateTime import DateTime
from processImage import StarImageProcess
from processData import ProcessData


def change_angles(database, starId, imageTime):
    database = np.array(database)
    index = np.where(database[:, 0] == starId)
    star = database[index, :]
    date = DateTime(imageTime)
    # calculation longitude
    t_from_days = date.numDays / 36525
    phi0 = (100.4606 + 36000.7701 * t_from_days) % 360
    # print(star[0,0,1] * 180 / 2 / np.pi)
    hours = (date.hour + date.minute / 60 + date.second / 60 / 60) / 24 * 360  # time in degrees
    alpha = star[0, 0, 1] * 180 / 2 / np.pi
    delta = star[0, 0, 2]*180/2/np.pi
    DeltaAlpha = 3.075 + 1.336 * np.sin(alpha) * np.tan(delta)
    DeltaDelta = 20.04 * np.cos(alpha)
    n = date.year - 2000
    alpha += n * DeltaAlpha
    delta += n * DeltaDelta
    alpha = alpha
    delta = delta
    longitude = (phi0 - alpha + (1.0027 * hours)) % 360
    print("Longitude: ", longitude)
    latitude = delta % 90
    print("Latitude: ", latitude)
    return latitude, longitude


def find_localization(path="../pictures/IMG_9527.jpg", databasePath="../database/hygdata_v3.csv"):
    localizations = StarImageProcess().find_stars(path=path)
    p = ProcessData()
    database = p.read_and_process_database(databasePath)
    ids = p.search_most_similar_stars(localizations, database, databasePath)
    print("final ids: ", ids)
    # [86403.0, 39644.0, 107213.0]
    # [86403.0, 39644.0, 107213.0]
    # [40922.0, 31052.0, 57408.0]
    # [108086.0, 63153.0, 65433.0]
    # [22400.0, 19727.0, 20661.0] the most promising
    # [64939.0, 81859.0, 72868.0, 62131.0, 72028.0]
    counter=0
    sumlat=0
    sumlong=0
    for currentid in ids:
        latitude, longitude = change_angles(database, currentid, "26 08 2018 23:05:46")
        sumlat += latitude
        sumlong += longitude
        counter += 1
    print(sumlat/counter, sumlong/counter)

    # change_angles(database, 72028, "26 08 2018 23:26:12")


find_localization()
