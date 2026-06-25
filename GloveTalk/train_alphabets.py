import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

CSV_FILENAME = "alphabet_dataset.csv"
MODEL_FILENAME = "alphabet_model.pkl"

print("Loading Alphabet Dataset...")
try:
    df = pd.read_csv(CSV_FILENAME)
except FileNotFoundError:
    print(f"Error: {CSV_FILENAME} not found. Collect some data first!")
    exit()

# 1. Separate the Labels (Y) from the Sensor Data (X)
# The label is the first column, the sensor data is everything else
y = df['label'].values
X = df.drop(columns=['label']).values

# 2. Split into Training (80%) and Testing (20%) sets
# This ensures the model is tested on data it has NEVER seen before
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Found {len(df)} total samples. Training on {len(X_train)}, Testing on {len(X_test)}.")

# 3. Build and Train the Random Forest
print("Training the Random Forest AI...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Test the Accuracy
print("Testing accuracy...")
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\n✅ AI Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, predictions))

# 5. Save the trained brain to your hard drive
joblib.dump(model, MODEL_FILENAME)
print(f"\nModel successfully saved as '{MODEL_FILENAME}'!")