# Import libraries
from pandas import read_csv
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the csv file using read_csv function of pandas library
myFilename = 'TrainingRoadCrashData.csv'

# Define the data variable names
myNames = ['Month', 'Season', 'WeekDay','DaySpan', 'Light', 'Weather', 'Traffic', 'Surface', 'Division',
        'TowFactor', 'HeavyVehicle', 'LGA/Area', 'Rural/Urban', 'SpeedRelated', 'RoadClass']

# read dataset
myDataframe = read_csv(myFilename, names=myNames)

# Retrieve explanatory variable column names 
myExplanatoryvariables = myDataframe.columns[0:14]

# Retrieva explanatory variables values
myExplanatoryvalues = myDataframe.loc[:,myExplanatoryvariables]

# Convert the explanatory variables into a matrix
myExplanatorymatrix = myExplanatoryvalues.iloc[:,:].values

# Identify outliers based on explantory variable values (-1: outliers, 1: inliers)
myIso = IsolationForest(contamination=0.1, random_state=0)
myForest = myIso.fit_predict(myExplanatorymatrix)

# Add a new column next to the explantory variable indicating whether a row is an outlier (-1) or not (1)
myExplanatoryvalues['Outlier'] = myForest

# Get rows which are outliers (-1)
myOutliers = myExplanatoryvalues.loc[myExplanatoryvalues['Outlier']==-1]

# Extract indices of the outlying rows
myOutlierindex = list(myOutliers.index)

# Apply Standardisation
myScaler = StandardScaler()
myScaledvalues = myScaler.fit_transform(myExplanatorymatrix)

# Use PCA to reduce the 14-variable data to 3 components
myPCA = PCA(n_components=2)
myPrincipalcomponents = myPCA.fit_transform(myScaledvalues)

# Cast the PCA values as a DataFrame
myDataframe2D = pd.DataFrame(myPrincipalcomponents)

# Plot figure
myFigure = plt.figure(figsize=(10,10))
plt.title('Road Crash 2D Outlier Detection with Standardisation')

# Plot inliers
plt.scatter(myDataframe2D[0], myDataframe2D[1], s=50, lw=1, label="Inliers",c="green")

# Plot outliers
plt.scatter(myDataframe2D.iloc[myOutlierindex,0], myDataframe2D.iloc[myOutlierindex,1], lw=2, s=50, c="red", label="Outliers")

# Add figure legend and data labels
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend(loc="upper right")

# Print plot
plt.show()