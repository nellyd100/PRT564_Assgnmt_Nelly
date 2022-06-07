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

# read dataset
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

# Print the accuracy of Linear Discriminant
print()
print('Mean Estimated Accuracy Linear Discriminant: %f ' % (myResult.mean()))
print()

# Define one new data instance for prediction
myNewexplanatoryvariables = [[0,0,1,70,7,10,2,5,1,1,6,1,4,99,2,3194,21,3,6,5,2,2,1],
                             [700,0,8,70,7,10,2,4,5,1,6,1,9,99,4,1464,3,1,6,1,2,2,1],
                             [0,0,1,70,7,10,2,5,1,1,6,1,4,99,2,3194,21,3,6,5,2,2,1],
                             [0,0,1,70,7,10,2,5,1,1,13,2,6,99,2,3194,21,3,6,5,2,2,1],
                             [0,0,1,73,2,10,1,7,1,1,6,1,4,99,2,1215,21,2,6,1,2,2,1]]
        
# Predict outcome
myNewresponsivevariable = myFitmodel.predict(myNewexplanatoryvariables)
print("Input Row 1 =%s, Predicted =%s" % (myNewexplanatoryvariables[0], myNewresponsivevariable[0]))
print("Input Row 2 =%s, Predicted =%s" % (myNewexplanatoryvariables[1], myNewresponsivevariable[1]))
print("Input Row 3 =%s, Predicted =%s" % (myNewexplanatoryvariables[2], myNewresponsivevariable[2]))
print("Input Row 4 =%s, Predicted =%s" % (myNewexplanatoryvariables[3], myNewresponsivevariable[3]))
print("Input Row 5 =%s, Predicted =%s" % (myNewexplanatoryvariables[4], myNewresponsivevariable[4]))
print()