import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report # <-- Added for the detailed accuracy report

CSV_FILENAME = "sign_language_dataset.csv"
MODEL_FILENAME = "verb_model.h5"
TARGET_FRAMES = 50  # Matches your updated collection script!
NUM_FEATURES = 18   # 18 sensor columns total

print("Loading Dynamic Verb Dataset...")
df = pd.read_csv(CSV_FILENAME)

# Separate the raw data
sensor_data = df.drop(columns=['timestamp', 'label']).values
labels = df['label'].values

# Calculate how many full 50-frame "Takes" we have
num_samples = len(df) // TARGET_FRAMES

# Reshape the flat CSV into 3D blocks: [Number of Takes, 50 Timesteps, 18 Sensors]
X = sensor_data[:num_samples * TARGET_FRAMES].reshape((num_samples, TARGET_FRAMES, NUM_FEATURES))

# Grab one label for every full take, stopping exactly at the truncated cutoff
y_raw = labels[:num_samples * TARGET_FRAMES:TARGET_FRAMES]

# Convert text labels ("hello", "thanks") into neural network numbers (0, 1)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_raw)
y_categorical = to_categorical(y_encoded)
num_classes = len(encoder.classes_)

print(f"Found {num_samples} dynamic signs across {num_classes} classes: {encoder.classes_}")

# Split into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Build the LSTM Brain
print("Building LSTM Neural Network...")
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(TARGET_FRAMES, NUM_FEATURES)))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the network
print("\nStarting Training (Epochs)...")
model.fit(X_train, y_train, epochs=40, batch_size=8, validation_data=(X_test, y_test))

# ==========================================
# ACCURACY CHECKING BLOCK
# ==========================================
print("\n📊 Evaluating Final Model Accuracy...")

# 1. Get the overall percentage score
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ OVERALL TEST ACCURACY: {accuracy * 100:.2f}%\n")

# 2. Get a detailed breakdown of every single word
print("🔍 Word-by-Word Performance Report:")
predictions = model.predict(X_test, verbose=0)

# Convert the AI's probability scores back into simple class numbers (0, 1, 2...)
y_pred_classes = np.argmax(predictions, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Print the report
print(classification_report(y_true_classes, y_pred_classes, target_names=encoder.classes_, labels=np.arange(num_classes), zero_division=0))
# ==========================================

# Save the model and the label dictionary
model.save(MODEL_FILENAME)
np.save('verb_classes.npy', encoder.classes_)
print(f"\n✅ Model successfully saved as '{MODEL_FILENAME}'!")