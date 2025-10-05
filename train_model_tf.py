# train_model_tf.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import os

ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, "data", "city_day.csv")
MODEL_PATH = os.path.join(ROOT, "model", "aqi_tf.h5")
SCALER_PATH = os.path.join(ROOT, "model", "scaler.pkl")

# Load data
df = pd.read_csv(DATA_PATH)
df = df.dropna()

POLLUTANTS = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene']
X = df[POLLUTANTS].values
y = df['AQI'].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_PATH)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build TF model
model = Sequential([
    Dense(64, activation='relu', input_shape=(len(POLLUTANTS),)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=1)

# Save model
model.save(MODEL_PATH)

print("✅ Model trained and saved successfully.")
