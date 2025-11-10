import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("uber.csv").dropna()

if 'Date/Time' in df.columns:
    df['Date/Time'] = pd.to_datetime(df['Date/Time'], errors='coerce')
    df['hour'] = df['Date/Time'].dt.hour
    df['day'] = df['Date/Time'].dt.day
    df['month'] = df['Date/Time'].dt.month

for col in ['key', 'id', 'Unnamed: 0']:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

q1, q3 = df['fare_amount'].quantile([0.25, 0.75])
IQR = q3 - q1
df = df[(df['fare_amount'] >= q1 - 1.5 * IQR) & (df['fare_amount'] <= q3 + 1.5 * IQR)]

sns.heatmap(df.corr(), annot=True, cmap="Blues")
plt.show()

X = df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']]
y = df['fare_amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)

def evaluate(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name}: R2={r2:.3f}, RMSE={rmse:.3f}")

evaluate("Linear Regression", y_test, y_pred_lr)
evaluate("Random Forest", y_test, y_pred_rf)
