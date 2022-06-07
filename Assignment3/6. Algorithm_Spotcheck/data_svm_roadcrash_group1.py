# Import libraries
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# Load the csv file using read_csv function of pandas library
myFilename = 'TrainingRoadCrashInliersData.csv'

# Define the data variable names
myNames = ['Mcycle', 'NbrTow', 'Airbag', 'VehVMC1', 'LicStatus', 'ABSSA3', 'Aboriginal', 'AgeBand', 'SafDevice',
           'Sex', 'State1', 'State2', 'RoadUser1', 'RoadWidth', 'TrafDensity', 'ATVInv', 'UnitType', 'VehDirTravel',
           'VehMov', 'VertFeature', 'RigidVeh', 'Businv', 'Pedestrian', 'InjuryDesc']

# Read dataset
myDataframe = read_csv(myFilename, names=myNames)

# Extract the dataframe values
myArray = myDataframe.values

# Split the array into input and output
myExplanatoryvariables = myArray[:,0:23]
myResponsivevariable = myArray[:,23]

# Setup evaluation algorithm
myFolds = 10
myRandomseed = 0
myValidation = KFold(n_splits = myFolds, random_state = myRandomseed, shuffle = True)
myModel = SVC(gamma='auto')
myResults = cross_val_score(myModel, myExplanatoryvariables, myResponsivevariable, cv=myValidation)

# Print the accuracy of SVM
print()
print('Mean Estimated Accuracy SVM: %f ' % (myResults.mean()))
print()