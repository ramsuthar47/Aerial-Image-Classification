import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "Dataset", "raw")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "Dataset", "processed", "aerial_data.npz")
IMAGE_SIZE = (128, 128)

def load_images_from_folder(folder):
    images = []
    labels = []
    class_names = sorted(os.listdir(folder))
    print(f"Detected classes: {class_names}")

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(folder, class_name)
        if not os.path.isdir(class_dir):
            continue

        image_files = []
        for root, _, files in os.walk(class_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    image_files.append(os.path.join(root, f))

        for img_path in image_files:
            try:
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
                img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    print(f"Loaded {len(images)} images across {len(class_names)} classes.")
    return np.array(images), np.array(labels), class_names

def main():
    print("Loading raw images...")
    X, y, class_names = load_images_from_folder(RAW_DATA_PATH)

    if len(X) == 0:
        raise ValueError("No images were loaded. Check your Dataset/raw folder structure.")

    print("Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    np.savez_compressed(
        PROCESSED_DATA_PATH,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        class_names=class_names
    )

    print(f"Processed dataset saved at: {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    main()