import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "Dataset", "processed", "aerial_data.npz")
FEATURES_OUTPUT_PATH = os.path.join(BASE_DIR, "Dataset", "features", "vgg16_features.npz")

os.makedirs(os.path.dirname(FEATURES_OUTPUT_PATH), exist_ok=True)


def load_processed_data():
    """Load preprocessed image data and labels."""
    data = np.load(PROCESSED_DATA_PATH, allow_pickle=True)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    class_names = data["class_names"]
    return X_train, X_test, y_train, y_test, class_names


def extract_features(model, data, batch_size=32):
    """Extract deep features using pretrained CNN."""
    features_list = []
    for i in tqdm(range(0, len(data), batch_size), desc="Extracting features"):
        batch = data[i:i + batch_size]
        batch_preprocessed = tf.keras.applications.vgg16.preprocess_input(batch)
        features = model.predict(batch_preprocessed, verbose=0)
        features_list.append(features)
    return np.vstack(features_list)


def main():
    print("Loading processed data...")
    X_train, X_test, y_train, y_test, class_names = load_processed_data()
    print(f"Loaded processed data: {X_train.shape[0]} train, {X_test.shape[0]} test images.")

    print("Loading pretrained VGG16 model (without top layers)...")
    base_model = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
    model = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)

    print("Extracting deep features for training data...")
    train_features = extract_features(model, X_train)
    print("Extracting deep features for testing data...")
    test_features = extract_features(model, X_test)

    print("Saving extracted features...")
    np.savez_compressed(
        FEATURES_OUTPUT_PATH,
        train_features=train_features,
        test_features=test_features,
        y_train=y_train,
        y_test=y_test,
        class_names=class_names,
    )

    print(f"Feature extraction complete. Saved at: {FEATURES_OUTPUT_PATH}")


if __name__ == "__main__":
    main()