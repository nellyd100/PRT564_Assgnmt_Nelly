# Import libraries
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# Load the csv file using read_csv function of pandas library
myFilename = 'TrainingDriverBehaviourInliersData.csv'

# Define the data variable names
myNames = ['Experience', 'KlmsDist', 'Meter', 'Occupants', 'VehSeq', 'Airbag', 'AlcoholRel', 'LicClass', 'PedVMC',
           'HeavyVeh', 'Carriage', 'DUI', '4WD', 'Intername', 'LicStatus', 'NTRes', 'Community', 'Country', 'SafDevice',
           'Sex', 'State1', 'RegoState', 'RoadDivision', 'RoadName', 'RoadUser1', 'RoadWidth', 'Rural', 'SpeedRel',
           'SurfaceType', 'TSD', 'UnitType', 'VehMov', 'VehVMC', 'VertFeature', 'Weather', 'CyclistInv', 'ArticVeh1',
           'RigidVeh', 'ArticVeh2', 'ContFactor']

# Read dataset
myDataframe = read_csv(myFilename, names=myNames)

# Extract the dataframe values
myArray = myDataframe.values

# Split the array into input and output
myExplanatoryvariables = myArray[:,0:39]
myResponsivevariable = myArray[:,39]

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