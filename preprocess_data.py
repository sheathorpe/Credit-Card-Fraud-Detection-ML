from itertools import groupby

import pandas
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def transform_data(df):
    transform_transaction_date(df)
    transform_amount(df)
    compute_fraud_rate_for_merchant_id(df)

    # df = df.drop(['TransactionDate', 'Hour', 'Amount'], axis='columns')  # todo: drop at end bc it makes copies

def transform_transaction_date(df):
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])

    time_bins = [0, 6, 12, 18, 24]
    time_labels = ['Late Night', 'Morning', 'Afternoon', 'Night']

    df['Hour'] = df['TransactionDate'].dt.hour
    df['TimeOfDay'] = pd.cut(x=df['Hour'], bins=time_bins, labels=time_labels)

    df['DayOfWeek'] = df['TransactionDate'].dt.day_name()

def transform_amount(df):
    scaler = RobustScaler()
    # WARNING: If we add more columns we have to do it before normalization
    scaler.fit(df[['Amount']])
    df['AmountScaled'] = scaler.transform(df[['Amount']])

# todo: we would need to only calculate this only for the training data, then map to testing data to avoid leakage
# when mapping the the default value should be the avg global fraud rate for dataset
def compute_fraud_rate_for_merchant_id(df):
    # todo: get global baseline which we will use for test
    fraud_rates_by_merchant_id = df.groupby('MerchantID')['IsFraud'].mean()
    df['MerchantFraudRate'] = df['MerchantID'].map(fraud_rates_by_merchant_id)

'''
There are 10 unique cities in our dataset and each merchant id has at least one location in each city
Will create a pseudo-unique identifier for transactions on MerchantID, Location, Amount
'''
def pair_transactions(df):
    print(df.groupby(['MerchantID', 'Location'])['Amount'].nunique())

data_df = pd.read_csv('data/data.csv')
transform_data(data_df)
pair_transactions(data_df)
print(data_df.head())