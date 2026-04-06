import pandas as pd
import numpy as np

acsdata = "https://api.census.gov/data/2024/acs/acs5?get=NAME,B19013_001E,B17001_001E,B17001_002E,B19083_001E,B25003_001E,B25003_003E,B23025_003E,B23025_005E&for=metropolitan%20statistical%20area/micropolitan%20statistical%20area:*"
censusdf = pd.read_json(acsdata)

censusdf.columns = censusdf.iloc[0]
censusdf = censusdf[1:]

acsvariables = ["B19013_001E", "B17001_001E", "B17001_002E", "B19083_001E", "B25003_001E", "B25003_003E", "B23025_003E", "B23025_005E"]
for v in acsvariables:
    censusdf[v] = pd.to_numeric(censusdf[v])

censusdf.rename(columns={
    'B19013_001E': 'medianIncome',
    "B17001_001E": "totalPopulation",
    "B17001_002E": "belowPoverty",
    "B19083_001E": "giniIndex",
    "B25003_001E": "housedPopulation",
    "B25003_003E": "renters",
    "B23025_003E": "totalLaborForce",
    "B23025_005E": "laborForceUnemployed"
})

censusdf.columns = ['name', 'medianIncome', 'totalPopulation', 'belowPoverty', 'giniIndex',
 'housedPopulation', 'renters', 'totalLaborForce', 'laborForceUnemployed', 'metroArea']

#print(censusdf[1:5])
metroAreas = ['Chicago-Naperville-Elgin, IL-IN Metro Area', 
            'Dallas-Fort Worth-Arlington, TX Metro Area',
            'Houston-Pasadena-The Woodlands, TX Metro Area',
            'Los Angeles-Long Beach-Anaheim, CA Metro Area',
            'New York-Newark-Jersey City, NY-NJ Metro Area',
            'Philadelphia-Camden-Wilmington, PA-NJ-DE-MD Metro Area',
            'Phoenix-Mesa-Chandler, AZ Metro Area',
            'San Antonio-New Braunfels, TX Metro Area',
            'San Diego-Chula Vista-Carlsbad, CA Metro Area',
            'San Jose-Sunnyvale-Santa Clara, CA Metro Area']

#print(censusdf[censusdf['name'].isin(metroAreas)])

frauddf = censusdf[censusdf['name'].isin(metroAreas)]

frauddf = frauddf.drop(columns=['metroArea'])

frauddf['povertyRate'] = frauddf['belowPoverty'] / frauddf['totalPopulation']
frauddf = frauddf.drop(columns=['belowPoverty', 'totalPopulation'])

frauddf['renterPercentage'] = frauddf['renters'] / frauddf['housedPopulation']
frauddf = frauddf.drop(columns=['renters', 'housedPopulation'])

frauddf['unemploymentRate'] = frauddf['laborForceUnemployed'] / frauddf['totalLaborForce']
frauddf = frauddf.drop(columns=['laborForceUnemployed', 'totalLaborForce'])

print(frauddf)



