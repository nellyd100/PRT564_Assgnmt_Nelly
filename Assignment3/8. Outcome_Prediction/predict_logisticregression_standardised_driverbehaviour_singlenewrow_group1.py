# Import functions
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Load the csv file using read_csv function of pandas library
myFilename = 'TrainingDriverBehaviourInliersData.csv'

# Define the data variable names
myNames = ['Experience', 'KlmsDist', 'Meter', 'Occupants', 'VehSeq', 'Airbag', 'AlcoholRel', 'LicClass', 'PedVMC',
           'HeavyVeh', 'Carriage', 'DUI', '4WD', 'Intername', 'LicStatus', 'NTRes', 'Community', 'Country', 'SafDevice',
           'Sex', 'State1', 'RegoState', 'RoadDivision', 'RoadName', 'RoadUser1', 'RoadWidth', 'Rural', 'SpeedRel',
           'SurfaceType', 'TSD', 'UnitType', 'VehMov', 'VehVMC', 'VertFeature', 'Weather', 'CyclistInv', 'ArticVeh1',
           'RigidVeh', 'ArticVeh2', 'ContFactor']

# read dataset
myDataframe = read_csv(myFilename, names=myNames)

# Extract the dataframe values
myArray = myDataframe.values

# Split the array into input and output
myExplanatoryvariables = myArray[:,0:39]
myResponsivevariable = myArray[:,39]

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

# Define one new data instance for prediction
myNewexplanatoryvariables = [[99,0,0,1,3,2,2,1,21,1,1,2,1,1266,7,1,206,39,11,2,6,6,1,1824,4,99,2,2,1,5,21,18,7,1,1,2,2,2,2]]

# Predict outcome
myNewresponsivevariable = myFitmodel.predict(myNewexplanatoryvariables)
print()
myNewresponsivevariableprobability = myFitmodel.predict_proba(myNewexplanatoryvariables)
print()
print("Input Row 1 =%s, Predicted =%s" % (myNewexplanatoryvariables[0], myNewresponsivevariable[0]))
print()