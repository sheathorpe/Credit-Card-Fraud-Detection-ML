from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
import pandas as pd
from preprocessing import transform_data

data_df = pd.read_csv('data/data.csv')
x_train, y_train, x_test, y_test = transform_data(data_df)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)

model.fit(X=x_train, y=y_train)
y_predict = model.predict(x_test)
y_probs = model.predict_proba(x_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_probs)
recall = recall_score(y_test, y_predict)
prec = precision_score(y_test, y_predict)

print(f'ROC AUC: {roc_auc}')
print(f'Recall: {recall}')
print(f'Precision: {prec:}')