# Import libraries
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Load the csv file using read_csv function of pandas library
myFilename = 'AllVariablesTrainingRoadCrashInliersData.csv'

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
          'InjuryDesc']

# Read dataset
myDataframe = read_csv(myFilename, names=myNames)

# Extract the dataframe values
myArray = myDataframe.values

# Split the array into input and output
myExplanatoryvariables = myArray[:,0:119]
myResponsivevariable = myArray[:,119]

# Apply Standardisation
##myScaler = StandardScaler().fit(myExplanatoryvariables)
#myStandardisedexplanatoryvariables = myScaler.fit_transform(myExplanatoryvariables)

# Select the 14 most important features
myTest = SelectKBest(score_func=chi2, k=25)
myFitmodel = myTest.fit(myExplanatoryvariables, myResponsivevariable)

# Print the scores for the features
set_printoptions(precision=3)
print()
print(myFitmodel.scores_)
print()

# Select the variables to be printed
myFeatures = myFitmodel.transform(myExplanatoryvariables)

# Print the first five rows of the best 14 features (Columns) selected
print(myFeatures[0:10,:])
print()