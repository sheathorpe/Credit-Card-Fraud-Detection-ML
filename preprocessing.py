import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN, MiniBatchKMeans

scaler = RobustScaler()

time_encoding = {'Late Night': 0, 'Morning': 1, 'Afternoon': 2, 'Night': 3}

weekday_encoding = {
    'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
    'Thursday': 4, 'Friday': 5, 'Saturday': 6
}

transaction_type_encoding = {'purchase': 0, 'refund': 1}

location_encoding = {
    'San Antonio': 0, 'Dallas': 1, 'New York': 2, 'Philadelphia': 3,
    'Phoenix': 4, 'Chicago': 5, 'San Jose': 6, 'San Diego': 7, 'Houston': 8, 'Los Angeles': 9
}

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

def transform_data(df):
    transform_transaction_date(df)

    df = merge_census_data_with_fraud_data(df)

    df['TimeOfDay'] = df['TimeOfDay'].map(time_encoding)
    df['DayOfWeek'] = df['DayOfWeek'].map(weekday_encoding)
    df['TransactionType'] = df['TransactionType'].map(transaction_type_encoding)
    df['Location'] = df['Location'].map(location_encoding)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=0, stratify=df['IsFraud'])
    compute_fraud_rate_for_merchant_id(train_df, test_df)
    transform_amount(train_df, test_df)
    perform_clustering(train_df, test_df)

    columns_to_drop = ['TransactionDate', 'Hour', 'Amount', 'TransactionID', 'MerchantID', 'name']

    test_df.drop(
        columns_to_drop, axis='columns', inplace=True
    )

    train_df.drop(
        columns_to_drop, axis='columns', inplace=True
    )

    test_df.dropna(axis='rows', inplace=True)
    train_df.dropna(axis='rows', inplace=True)

    x_train = train_df.drop('IsFraud', axis=1)
    y_train = train_df['IsFraud']

    x_test = test_df.drop('IsFraud', axis=1)
    y_test = test_df['IsFraud']

    return x_train, y_train, x_test, y_test

def transform_transaction_date(df):
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])

    time_bins = [0, 6, 12, 18, 24]
    time_labels = ['Late Night', 'Morning', 'Afternoon', 'Night']

    df['Hour'] = df['TransactionDate'].dt.hour
    df['TimeOfDay'] = pd.cut(x=df['Hour'], bins=time_bins, labels=time_labels, include_lowest=True)
    df['DayOfWeek'] = df['TransactionDate'].dt.day_name()

def transform_amount(train_df, test_df):
    scaler.fit(train_df[['Amount']])

    train_df['AmountScaled'] = scaler.transform(train_df[['Amount']])
    test_df['AmountScaled'] = scaler.transform(test_df[['Amount']])

def compute_fraud_rate_for_merchant_id(train_df, test_df):
    avg_fraud = train_df['IsFraud'].mean()
    fraud_rates_by_merchant_id = train_df.groupby('MerchantID')['IsFraud'].mean()

    train_df['MerchantFraudRate'] = train_df['MerchantID'].map(fraud_rates_by_merchant_id)
    test_df['MerchantFraudRate'] = test_df['MerchantID'].map(fraud_rates_by_merchant_id)
    test_df['MerchantFraudRate'] = test_df['MerchantFraudRate'].fillna(avg_fraud)

def load_census_data():
    census_df = pd.read_json('data/acs5.json')
    census_df.columns = census_df.iloc[0]
    census_df = census_df[1:]

    acs_variables = ["B19013_001E", "B17001_001E", "B17001_002E", "B19083_001E",
                     "B25003_001E", "B25003_003E", "B23025_003E", "B23025_005E"]

    for v in acs_variables:
        census_df[v] = pd.to_numeric(census_df[v])

    census_df.rename(columns={
        "NAME": "name",
        'B19013_001E': 'medianIncome',
        "B17001_001E": "totalPopulation",
        "B17001_002E": "belowPoverty",
        "B19083_001E": "giniIndex",
        "B25003_001E": "housedPopulation",
        "B25003_003E": "renters",
        "B23025_003E": "totalLaborForce",
        "B23025_005E": "laborForceUnemployed",
        'metropolitan statistical area/micropolitan statistical area': 'metroArea'
    }, inplace=True)

    fraud_df = census_df[census_df['name'].isin(metroAreas)]
    fraud_df = fraud_df.drop(columns=['metroArea'])

    fraud_df['povertyRate'] = fraud_df['belowPoverty'] / fraud_df['totalPopulation']
    fraud_df = fraud_df.drop(columns=['belowPoverty', 'totalPopulation'])

    fraud_df['renterPercentage'] = fraud_df['renters'] / fraud_df['housedPopulation']
    fraud_df = fraud_df.drop(columns=['renters', 'housedPopulation'])

    fraud_df['unemploymentRate'] = fraud_df['laborForceUnemployed'] / fraud_df['totalLaborForce']
    fraud_df = fraud_df.drop(columns=['laborForceUnemployed', 'totalLaborForce'])

    fraud_df['name'] = fraud_df['name'].str.split('-').str[0]

    return fraud_df

def merge_census_data_with_fraud_data(fraud_df):
    census_df = load_census_data()

    fraud_df = pd.merge(
        fraud_df,
        census_df,
        left_on='Location',
        right_on='name',
        how='left'
    )

    return fraud_df

def perform_clustering(train_df, test_df):
    clustering_features = ['AmountScaled', 'MerchantFraudRate', 'unemploymentRate']

    # K-MEANS (This part was already perfect)
    kmeans = MiniBatchKMeans(n_clusters=5, random_state=0, n_init='auto')
    kmeans.fit(train_df[clustering_features])
    train_df['DistToClusterCenter'] = kmeans.transform(train_df[clustering_features]).min(axis=1)
    test_df['DistToClusterCenter'] = kmeans.transform(test_df[clustering_features]).min(axis=1)

    # DBSCAN SAMPLE
    sample_size = 50000
    sample_df = train_df.sample(n=sample_size, random_state=0)

    dbscan = DBSCAN(eps=0.5, min_samples=10)
    sample_cluster_labels = dbscan.fit_predict(sample_df[clustering_features])

    # 1. Get the noise labels JUST for the 10k sample
    sample_is_noise = (sample_cluster_labels == -1).astype(int)

    # 2. Train the KNN Surrogate on the 10k sample X and 10k sample y
    knn_surrogate = KNeighborsClassifier(n_neighbors=5)
    knn_surrogate.fit(sample_df[clustering_features], sample_is_noise)

    # 3. Predict the noise flag for the FULL 80k train set and 20k test set
    train_df['IsNoise'] = knn_surrogate.predict(train_df[clustering_features])
    test_df['IsNoise'] = knn_surrogate.predict(test_df[clustering_features])