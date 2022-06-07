# Input libraries
from pandas import read_csv
from pandas import set_option

# Load the csv file using read_csv function of pandas library
myFilename = 'RoadCrashData.csv'
myNames = ['Month', 'WeekDay','DaySpan', 'Light', 'Weather', 'Traffic', 'Surface', 'Division', 'TowFactor',
            'HeavyVehicle', 'LGA/Area', 'Rural/Urban', 'SpeedRelated', 'RoadClass']

# Read the csv file
myData = read_csv(myFilename, names=myNames)

# Set printing options
set_option('display.width', 250)
set_option('display.max_columns', 14)
set_option('precision', 3)

# Print dat correlations
myCorrelations = myData.corr(method='pearson')
print(myCorrelations)