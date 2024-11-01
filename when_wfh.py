import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

df = pd.read_excel("C:\\Users\\81701\\Desktop\\when_wfh.xlsx", engine='openpyxl')

df['date'] = pd.to_datetime(df['date'])

df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['week'] = df['date'].dt.isocalendar().week

df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
df = df.drop(columns=['weekday'])

X = df.drop(columns=['wfh', 'date']) 
y = df['wfh']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_predictions))
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_predictions))

future_dates = pd.date_range(start='2023-11-02', periods=30, freq='D')
future_df = pd.DataFrame({'date': future_dates})

future_df['day'] = future_df['date'].dt.day
future_df['month'] = future_df['date'].dt.month
future_df['week'] = future_df['date'].dt.isocalendar().week

future_df['weekday_sin'] = np.sin(2 * np.pi * future_df['date'].dt.weekday / 7)
future_df['weekday_cos'] = np.cos(2 * np.pi * future_df['date'].dt.weekday / 7)

missing_cols = set(X_train.columns) - set(future_df.columns)
for col in missing_cols:
    future_df[col] = 0  

future_predictions = rf_model.predict(future_df.drop(columns=['date']))

future_df['wfh_prediction'] = future_predictions

print(future_df[['date', 'wfh_prediction']])
