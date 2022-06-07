# Import libraries
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier

# Load the csv file using read_csv function of pandas library
myFilename = 'AllVariablesTrainingDriverBehaviourInliersData.csv'

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

# Extract the dataframe values
myArray = myDataframe.values

# Split the array into input and output
myExplanatoryvariables = myArray[:,0:119]
myResponsivevariable = myArray[:,119]

# Apply Standardisation
myScaler = StandardScaler().fit(myExplanatoryvariables)
myStandardisedexplanatoryvariables = myScaler.fit_transform(myExplanatoryvariables)

# Select the most important features
myModel = ExtraTreesClassifier(n_estimators=100)
myFitmodel = myModel.fit(myStandardisedexplanatoryvariables, myResponsivevariable)

# Print the first five rows of the best features (Columns) selected
print()
print(myFitmodel.feature_importances_)
print()