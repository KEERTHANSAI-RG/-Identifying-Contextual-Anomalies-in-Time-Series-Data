import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import os

print("PROGRAM STARTED")

# ----------------------------
# CHECK FILE PATH
# ----------------------------

file_path = "dataset/climate_data.csv"

print("Checking dataset file...")

if os.path.exists(file_path):
    print("Dataset found!")
else:
    print("ERROR: Dataset file not found!")
    exit()

# ----------------------------
# LOAD DATASET
# ----------------------------

print("Loading dataset...")

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")

# ----------------------------
# SHOW COLUMN NAMES
# ----------------------------

print("\nCOLUMN NAMES:")
print(df.columns)

# ----------------------------
# SHOW FIRST 5 ROWS
# ----------------------------

print("\nFIRST 5 ROWS:")
print(df.head())

# ----------------------------
# CONVERT DATE
# ----------------------------

print("\nConverting date column...")

df['date'] = pd.to_datetime(df['date'])

# ----------------------------
# CREATE CONTEXT FEATURES
# ----------------------------

print("Creating contextual features...")

df['month'] = df['date'].dt.month
df['season'] = (df['month'] % 12 // 3) + 1

# ----------------------------
# REMOVE NULL VALUES
# ----------------------------

print("Removing missing values...")

df = df.dropna()

# ----------------------------
# SELECT FEATURES
# ----------------------------

print("Selecting features...")

features = [
    'meantemp',
    'humidity',
    'wind_speed',
    'meanpressure',
    'month',
    'season'
]

X = df[features]

print("Features selected!")

# ----------------------------
# TRAIN MODEL
# ----------------------------

print("Training Isolation Forest model...")

model = IsolationForest(
    contamination=0.02,
    random_state=42
)

model.fit(X)

print("Model trained successfully!")

# ----------------------------
# DETECT ANOMALIES
# ----------------------------

print("Detecting anomalies...")

df['anomaly'] = model.predict(X)

df['anomaly'] = df['anomaly'].map({
    1: 0,
    -1: 1
})

# ----------------------------
# EXTRACT ANOMALIES
# ----------------------------

anomalies = df[df['anomaly'] == 1]

print("\nTOTAL ANOMALIES:")
print(len(anomalies))

# ----------------------------
# SAVE CSV
# ----------------------------

print("\nSaving anomaly CSV file...")

anomalies.to_csv(
    "output/detected_anomalies.csv",
    index=False
)

print("CSV SAVED SUCCESSFULLY!")

# ----------------------------
# GRAPH 1
# TEMPERATURE ANOMALY GRAPH
# ----------------------------

print("\nCreating Temperature Anomaly Graph...")

plt.figure(figsize=(15,6))

plt.plot(
    df['date'],
    df['meantemp'],
    label='Temperature'
)

plt.scatter(
    anomalies['date'],
    anomalies['meantemp'],
    color='red',
    label='Anomaly'
)

plt.title("Temperature Contextual Anomalies")
plt.xlabel("Date")
plt.ylabel("Mean Temperature")
plt.legend()

plt.savefig("output/temperature_anomaly_graph.png")

plt.close()

print("Temperature graph saved!")

# ----------------------------
# GRAPH 2
# HUMIDITY ANOMALY GRAPH
# ----------------------------

print("\nCreating Humidity Graph...")

plt.figure(figsize=(15,6))

plt.plot(
    df['date'],
    df['humidity'],
    label='Humidity'
)

plt.scatter(
    anomalies['date'],
    anomalies['humidity'],
    color='red',
    label='Anomaly'
)

plt.title("Humidity Contextual Anomalies")
plt.xlabel("Date")
plt.ylabel("Humidity")
plt.legend()

plt.savefig("output/humidity_anomaly_graph.png")

plt.close()

print("Humidity graph saved!")

# ----------------------------
# GRAPH 3
# MONTHLY TEMPERATURE TREND
# ----------------------------

print("\nCreating Monthly Trend Graph...")

monthly_avg = df.groupby('month')['meantemp'].mean()

plt.figure(figsize=(10,5))

monthly_avg.plot(marker='o')

plt.title("Monthly Average Temperature Trend")
plt.xlabel("Month")
plt.ylabel("Average Temperature")

plt.savefig("output/monthly_temperature_trend.png")

plt.close()

print("Monthly trend graph saved!")

# ----------------------------
# GRAPH 4
# ANOMALY DISTRIBUTION
# ----------------------------

print("\nCreating Anomaly Distribution Graph...")

anomaly_counts = df['anomaly'].value_counts()

labels = ['Normal', 'Anomaly']

plt.figure(figsize=(6,6))

plt.pie(
    anomaly_counts,
    labels=labels,
    autopct='%1.1f%%'
)

plt.title("Anomaly Distribution")

plt.savefig("output/anomaly_distribution.png")

plt.close()

print("Distribution graph saved!")

# ----------------------------
# FINAL MESSAGE
# ----------------------------

print("\nPROJECT COMPLETED SUCCESSFULLY!")