import os
import numpy as np
import tensorflow as tf

DATASET_PATH = r"c:\Users\ramsu\Desktop\VS Projects\Aerial Image Classification\Dataset\processed\aerial_data.npz"
MODEL_PATH = r"c:\Users\ramsu\Desktop\VS Projects\Aerial Image Classification\models\vgg16_finetuned.keras"

print(f"Dataset path: {DATASET_PATH}")
print(f"Model will be saved to: {MODEL_PATH}")

data = np.load(DATASET_PATH)
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
class_names = data["class_names"]

print(f"Loaded dataset — Train: {X_train.shape}, Test: {X_test.shape}, Classes: {len(class_names)}")

if len(y_train.shape) == 1:
    print("Converting labels to one-hot encoding...")
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(class_names))
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(class_names))

num_classes = y_train.shape[1]

base_model = tf.keras.applications.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(128, 128, 3)
)
print(f"VGG16 base loaded with {len(base_model.layers)} layers.")

for layer in base_model.layers[:11]:
    layer.trainable = False
for layer in base_model.layers[11:]:
    layer.trainable = True

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=2, verbose=1
)

print("Starting fine-tuning training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

print("Evaluating fine-tuned model...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"Fine-tuned Test Accuracy: {test_acc * 100:.2f}%")
print(f"Fine-tuned model saved at: {MODEL_PATH}")