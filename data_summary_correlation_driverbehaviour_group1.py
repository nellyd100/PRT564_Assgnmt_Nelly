# Input libraries
from pandas import read_csv
from pandas import set_option

# Load the csv file using read_csv function of pandas library
myFilename = 'DriverBehaviourData.csv'
myNames = ['Occupants', 'Age', 'Sex', 'Aboriginal', 'Day', 'Alcohol', 'Drugs', 'Circumstance', 'RUMDesc', 
            'Pedestrian', 'Cyclist', 'DriverClass']

# Read the csv file
myData = read_csv(myFilename, names=myNames)

# Set printing options
set_option('display.width', 200)
set_option('display.max_columns', 12)
set_option('precision', 3)

# Print dat correlations
myCorrelations = myData.corr(method='pearson')
print(myCorrelations)