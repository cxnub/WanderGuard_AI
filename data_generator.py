from datetime import datetime
import os

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 10000

# Generate timestamps (over a 4 year period)
start_date = datetime(2020, 1, 1)
end_date = datetime(2024, 1, 1)
timestamps = pd.date_range(start_date, end_date, periods=num_samples)

# Generate synthetic features
data = {
    "timestamp": timestamps,
    "distance_from_safe_zone": np.where(
        np.random.rand(num_samples) < 0.3,  # 30% chance of being in safe zone
        0,  # If in safe zone, distance is 0
        np.random.randint(
            1, 500, num_samples
        ),  # Otherwise, generate a random distance between 1 and 500
    ),  # 30% chance of being in safe zone
    "heart_rate": np.random.randint(60, 120, num_samples),  # Heart rate in BPM
    "speed": np.random.uniform(0, 5, num_samples),  # Speed in meters per second
}

# Create DataFrame
df = pd.DataFrame(data)

# Adjust features when patient is in the safe zone
df.loc[df["distance_from_safe_zone"] == 0, "speed"] = np.random.uniform(
    0, 0.5, len(df[df["distance_from_safe_zone"] == 0])
)  # Low speed in safe zone
df.loc[df["distance_from_safe_zone"] == 0, "heart_rate"] = np.random.randint(
    60, 80, len(df[df["distance_from_safe_zone"] == 0])
)  # Normal heart rate in safe zone

# Add time-based features
df["hour_of_day"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.day_name()

# Assign scores to each rule
df['score'] = (
    (df['distance_from_safe_zone'] > 300).astype(int) +  # Rule 1
    (df['speed'] > 1.5).astype(int) +  # Rule 2
    (df['heart_rate'] > 80).astype(int) +  # Rule 3
    ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)).astype(int)  # Rule 4
)

# Classify as wandering if the score meets or exceeds a threshold
threshold = 2  # Adjust this threshold as needed
df['wandering_label'] = (df['score'] >= threshold).astype(int)

# drop unnecessary columns
df = df.drop(columns=['score', 'hour_of_day', 'day_of_week'])

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Display the first few rows
print(df.head())

# Save to CSV
os.makedirs("dataset", exist_ok=True)
df.to_csv("dataset/dementia_wandering_data.csv", index=False)
