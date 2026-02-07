import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\salon\OneDrive\Desktop\Tech Projects\Data_Science\Machine_learning\Codes_practice\Ensemble\creditcard.csv")
#df.info()
#df.describe()
#df.head(5)
#df['Class'].value_counts()

#Deviding into x and y variables

x = df.drop(columns = ['Class'], axis =1)
y = df['Class']



'''
x_resample , y_resample = smoteenn.fit_resample(x, y)

#lets check before and after resampling
print("Before:\n", y.value_counts())
print("After:\n", pd.Series(y_resample).value_counts())

#lets check shape
print("Original X shape:", x.shape)
print("Resampled X shape:", x_resample.shape)

print("Original y shape:", y.shape)
print("Resampled y shape:", y_resample.shape)

#now lets make a df which has resampled 

df_resampled = pd.concat(
    [pd.DataFrame(x_resample, columns=x.columns),
     pd.Series(y_resample, name='Class')],
    axis=1
)
'''

#spliting data
x_train, x_test, y_train, y_test = train_test_split(
    x,y, 
    test_size=0.2,
    random_state=42,
    stratify=y
)

## lets balance data
from imblearn.combine import SMOTEENN
smoteenn = SMOTEENN(random_state = 42)
x_resample , y_resample = smoteenn.fit_resample(x_train, y_train)

scaler = StandardScaler()
x_resample =scaler.fit_transform(x_resample)
x_test =scaler.transform(x_test)

from xgboost import XGBClassifier

model_xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    n_jobs=-1,
    random_state=42
)

from sklearn.metrics import accuracy_score, classification_report
model_xgb.fit(x_resample, y_resample)
y_pred_xgb = model_xgb.predict(x_test)
print("Accuracy (XGBoost):", accuracy_score(y_test, y_pred_xgb))
print("Classification Report (XGBoost):\n", classification_report(y_test, y_pred_xgb))


import joblib
joblib.dump(model_xgb, "xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")   # if used
joblib.dump(list(x.columns), "features.pkl")
