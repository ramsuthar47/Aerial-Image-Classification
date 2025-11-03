import os
import numpy as np
import tensorflow as tf
from keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "Dataset", "processed", "aerial_data.npz"))
MODEL_SAVE_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "models", "customcnn_fixed.keras"))

print(f"Dataset path: {DATASET_PATH}")
print(f"Model will be saved to: {MODEL_SAVE_PATH}")

data = np.load(DATASET_PATH, allow_pickle=True)
print(f"Keys in dataset: {list(data.keys())}")

X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
class_names = data["class_names"]

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
print(f"Number of classes: {len(class_names)}")

if y_train.ndim > 1 or y_train.max() > len(class_names) - 1:
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)
    print("Labels encoded with LabelEncoder")

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

model = models.Sequential([
    layers.Input(shape=X_train.shape[1:]),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=32,
    validation_data=(X_test, y_test)
)

model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")