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

# Retrieve explanatory variables values
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
myPCA = PCA(n_components=3)
myPrincipalcomponents = myPCA.fit_transform(myScaledvalues)

# Cast the PCA values as a DataFrame
myDataframe3D = pd.DataFrame(myPrincipalcomponents)

# Plot figure and subplots
myFigure = plt.figure(figsize=(10,10))
mySubplot = myFigure.add_subplot(111, projection='3d')

# Plot inliers
mySubplot.scatter(myDataframe3D[0], myDataframe3D[1], zs=myDataframe3D[2], s=50, lw=1, label="Inliers",c="green")

# Plot outliers
mySubplot.scatter(myDataframe3D.iloc[myOutlierindex,0], myDataframe3D.iloc[myOutlierindex,1],
                    myDataframe3D.iloc[myOutlierindex,2], lw=2, s=50, c="red", label="Outliers")

# Add figure legends and data labels
mySubplot.set_xlabel('Component 1')
mySubplot.set_ylabel('Component 2')
mySubplot.set_zlabel('Component 3')
mySubplot.set_title('Road Crash 3D Outlier Detection with Standardisation')
mySubplot.legend()

# Print plot
plt.show()