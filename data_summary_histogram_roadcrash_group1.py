# Input libraries
from matplotlib import pyplot
from pandas import read_csv

# Load the csv file using read_csv function of pandas library
myFilename = 'RoadCrashData.csv'
myNames = ['Month', 'WeekDay','DaySpan', 'Light', 'Weather', 'Traffic', 'Surface', 'Division', 'TowFactor',
            'HeavyVehicle', 'LGA/Area', 'Rural/Urban', 'SpeedRelated', 'RoadClass']

# Read the csv file
myData = read_csv(myFilename, names=myNames)

# Show the histogram
myData.hist(figsize=(10, 10))
pyplot.show()