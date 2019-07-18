import re
import numpy as np

class DateTime():
    def __init__(self, stringDate):
        segments = re.split(' |:', stringDate)
        self.day = int(segments[0])
        self.month = int(segments[1])
        self.year = int(segments[2])
        self.hour = int(segments[3])
        self.minute = int(segments[4])
        self.second = int(segments[5])
        self.numDays = 0
        self.calculate_days_from_2000()

    def calculate_days_from_2000(self):
        self.numDays = self.day
        self.numDays += (self.year - 2000) * 365 - np.floor((self.year - 2000)/4) + 1
        daysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self.numDays += sum(daysInMonth[0:self.month-1])
