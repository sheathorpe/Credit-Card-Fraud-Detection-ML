import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, roc_auc_score, accuracy_score
from preprocessing import transform_data

data_df = pd.read_csv('data/data.csv')
X_train, X_test, y_train, y_test = transform_data(data_df)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    # making the punishment for missing a fraud case is 5 times as high as for missing a genuine transaction
    class_weight={0:1, 1:5}
)

# synthetic minority over-sampling technique (SMOTE) injects artificial fraud cases into dataset
sm = SMOTE(random_state=42, sampling_strategy='auto')
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# training model on modified training/test sets
rf_model.fit(X=X_train_res, y=y_train_res)
y_predict = rf_model.predict(X_test)
y_probs = rf_model.predict_proba(X_test)[:, 1]

# using probabilities returned by model any probability greater than 10% is considered fraud
# this makes the model very vigilant
y_pred_new = (y_probs > 0.1).astype(int)

# calculating performance metrics
roc_auc = roc_auc_score(y_test, y_probs)
recall = recall_score(y_test, y_pred_new)
precision = precision_score(y_test, y_pred_new)
acc = accuracy_score(y_test, y_pred_new)

# printing metrics
print(f'Accuracy: {acc}')
print(f'ROC AUC: {roc_auc}')
print(f'Recall: {recall}')
print(f'Precision: {precision:}')