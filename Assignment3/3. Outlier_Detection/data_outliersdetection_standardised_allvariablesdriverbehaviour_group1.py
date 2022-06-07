# Import libraries
from pandas import read_csv
from sklearn.ensemble import IsolationForest

# Load the csv file using read_csv function of pandas library
myFilename = 'AllVariablesTrainingDriverBehaviourData.csv'

# Define the data variable names
myNames = ['Experience', 'KlmsDist', 'Mcycle', 'Meter', 'NbrInj', 'NbrTow', 'Occupants', 'VehSeq', 'AccDay', 'AccDesc',
          'AccTime', 'Acc10y', 'AccYear', 'Airbag', 'AlcoholRel', 'Circum1', 'Damage', 'Defects', 'LicClass', 'PedVMC',
          'Tow', 'TrafficControl', 'VehVMC1', 'VehVMC2','StreetFilter', 'Cyclist','Veh1TravelDir', 'DistPersons',
          'HeavyVeh', 'AccMonth', 'Carriage', 'Circum2', 'Accident', 'DirIDMark','DistanceID', 'DrugRel', 'AccVal',
          'DUI', '4WD', 'GeoCoded', 'Inspected', 'InsuInfo', 'TowAway','VehFire', 'Vetted', 'HitRun', 'HorFeature',
          'InjCode', 'InterName', 'LicStatus', 'LightCond', 'MinInjCode', 'MinInjDesc', 'Municipality', 'NTRes',
          'ABSSA3', 'LGA', 'NearID', 'NbrUnits', 'NbrRecords', 'OZTBefAft', 'OthFeature', 'Aboriginal', 'Ejected',
          'AgeBand', 'Age', 'Community', 'Country', 'SafDevice', 'Sex', 'State1', 'State2', 'PoliceArea',
          'PoliceDistrict', 'PoliceSup', 'RUMDesc', 'RegoExp', 'RegoState', 'RoadDivision', 'MainStFilter', 'RoadName',
          'RoadUser1', 'RoadUser2', 'RoadWidth', 'RdUserCode', 'Rural', 'SpeedLimit', 'SpeedRel', 'Sub/Area',
          'SurfaceType', 'TSD', 'TrafDensity', 'UnitMake', 'UnitModel', 'ATVInv', 'UnitTypeCode', 'UnitType',
          'UnitYear', 'VehDirTravel', 'VehMov', 'VehVMC', 'VertFeature', 'Weather', 'AccMthName', 'DistNbrCrash',
          'DARunTotal', 'CyclistInv', 'ArticVeh1', 'RigidVeh', 'BusInv', 'InterceptFilter', 'PointFound', 'ArticVeh2',
          'Veh6TravelDir', 'Veh5TravelDir', 'Veh4TravelDir', 'Veh2TravelDir', 'Veh3TravelDir', 'Pedestrian',
          'ContFactor']

# Read dataset
myDataframe = read_csv(myFilename, names=myNames)

# Retrieve explanatory variable column names 
myExplanatoryvariables = myDataframe.columns[0:119]

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

# Create a csv file with an outliers column
myExplanatoryvalues.to_csv('AllVariablesTrainingDriverBehaviourOutlierDetectionData.csv', index=None)