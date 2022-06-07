# Input libraries
from matplotlib import pyplot
from pandas import read_csv
import numpy

# Load the csv file using read_csv function of pandas library
myFilename = 'RoadCrashData.csv'
myNames = ['Month', 'WeekDay','DaySpan', 'Light', 'Weather', 'Traffic', 'Surface', 'Division', 'TowFactor',
            'HeavyVehicle', 'LGA/Area', 'Rural/Urban', 'SpeedRelated', 'RoadClass']

# Read the csv file
myData = read_csv(myFilename, names=myNames)

# Calculate correlation
myCorrelations = myData.corr()

# Plot correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(myCorrelations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,14,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(myNames)
ax.set_yticklabels(myNames)
pyplot.show()