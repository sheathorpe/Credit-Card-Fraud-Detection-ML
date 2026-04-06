import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

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

def transform_data(df):
    transform_transaction_date(df)
    transform_amount(df)

    df['TimeOfDay'] = df['TimeOfDay'].map(time_encoding)
    df['DayOfWeek'] = df['DayOfWeek'].map(weekday_encoding)
    df['TransactionType'] = df['TransactionType'].map(transaction_type_encoding)
    df['Location'] = df['Location'].map(location_encoding)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)
    compute_fraud_rate_for_merchant_id(train_df, test_df)

    columns_to_drop = ['TransactionDate', 'Hour', 'Amount', 'TransactionID', 'MerchantID']

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
    df['TimeOfDay'] = pd.cut(x=df['Hour'], bins=time_bins, labels=time_labels)

    df['DayOfWeek'] = df['TransactionDate'].dt.day_name()

def transform_amount(df):
    scaler.fit(df[['Amount']])
    df['AmountScaled'] = scaler.transform(df[['Amount']])

def compute_fraud_rate_for_merchant_id(train_df, test_df):
    avg_fraud = train_df['IsFraud'].mean()
    fraud_rates_by_merchant_id = train_df.groupby('MerchantID')['IsFraud'].mean()

    train_df['MerchantFraudRate'] = train_df['MerchantID'].map(fraud_rates_by_merchant_id)
    test_df['MerchantFraudRate'] = test_df['MerchantID'].map(fraud_rates_by_merchant_id)
    test_df['MerchantFraudRate'] = test_df['MerchantFraudRate'].fillna(avg_fraud)