import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

df = pd.read_excel("C:\\Users\\81701\\Desktop\\when_wfh.xlsx", engine='openpyxl')
df['date'] = pd.to_datetime(df['date'])

past_data = df.dropna(subset=['wfh'])
future_data = df[df['wfh'].isna()]

past_data['weekday_sin'] = np.sin(2 * np.pi * past_data['weekday'] / 7)
past_data['weekday_cos'] = np.cos(2 * np.pi * past_data['weekday'] / 7)
past_data = past_data.drop(columns=['weekday'])

X = past_data.drop(columns=['wfh', 'date'])
y = past_data['wfh']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_predictions))
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_predictions))

future_data['weekday_sin'] = np.sin(2 * np.pi * future_data['date'].dt.weekday / 7)
future_data['weekday_cos'] = np.cos(2 * np.pi * future_data['date'].dt.weekday / 7)
future_data = future_data.drop(columns=['weekday'])

missing_cols = set(X_train.columns) - set(future_data.columns)
for col in missing_cols:
    future_data[col] = 0

future_probabilities = rf_model.predict_proba(future_data.drop(columns=['date', 'wfh']))[:, 1]
future_data['wfh_probability'] = future_probabilities

print(future_data[['date', 'wfh_probability']])
