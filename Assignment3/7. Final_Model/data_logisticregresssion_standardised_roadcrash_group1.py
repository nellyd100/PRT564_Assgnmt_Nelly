# Import functions
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

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

# Apply Standardisation
myScaler = StandardScaler().fit(myExplanatoryvariables)
myStandardisedexplanatoryvariables = myScaler.fit_transform(myExplanatoryvariables)

# Setup evaluation algorithm
myFolds = 100
myRandomseed = 0
myValidation = KFold(n_splits = myFolds, random_state = myRandomseed, shuffle = True)
myModel = LogisticRegression(solver='liblinear', max_iter=10000)
myFitmodel = myModel.fit(myStandardisedexplanatoryvariables, myResponsivevariable)
myResult = cross_val_score(myModel, myStandardisedexplanatoryvariables, myResponsivevariable, cv=myValidation)

# Print the accuracy of Logistic Regression
print()
print('Mean Estimated Accuracy Logistic Regression: %f ' % (myResult.mean()))
print()