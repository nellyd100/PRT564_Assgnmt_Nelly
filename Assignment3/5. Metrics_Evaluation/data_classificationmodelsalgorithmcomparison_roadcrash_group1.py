# Import libraries
from pandas import read_csv
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
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

# Prepare models
myModel = []
myModel.append(('LR', LogisticRegression(solver='liblinear')))
myModel.append(('LDA', LinearDiscriminantAnalysis()))
myModel.append(('KNN', KNeighborsClassifier()))
myModel.append(('KM', KMeans()))
myModel.append(('CART', DecisionTreeClassifier()))
myModel.append(('NB', GaussianNB()))
myModel.append(('SVM', SVC(gamma='auto')))

# Evaluate each model in turn
myResult = []
myName = []
myRandomseed = 0
myScoring = 'accuracy'
for name, model in myModel:
    myValidation = KFold(n_splits=10, random_state = myRandomseed, shuffle = True)
    cv_results = cross_val_score(model, myExplanatoryvariables, myResponsivevariable, cv=myValidation, scoring=myScoring)
    myResult.append(cv_results)
    myName.append(name)
    myMessage = '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())
    print()
    print(myMessage)
    print()

# Plot figure and subplot
myFigure = plt.figure(figsize=(10,10))
mySubplot = myFigure.add_subplot(111)

# Add figure legend and data labels
mySubplot.set_xticklabels(myName)

# Add figure legends and data labels
mySubplot.set_xlabel('Algorithm')
mySubplot.set_ylabel('Accuracy Percentage')
mySubplot.set_title('Road Crash Classification Algorithm Comparison')

# Print boxplot algorithm comparison
plt.boxplot(myResult)
plt.show()