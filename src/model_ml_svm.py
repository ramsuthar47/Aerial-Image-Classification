import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

dataset_path = r"c:\Users\ramsu\Desktop\VS Projects\Aerial Image Classification\Dataset\processed\aerial_data.npz"
model_save_path = r"c:\Users\ramsu\Desktop\VS Projects\Aerial Image Classification\models\svm_fast.pkl"

data = np.load(dataset_path)
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
print(f"Number of classes: {len(np.unique(y_train))}")

X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)

pca_components = min(100, X_train.shape[1])  
pca = PCA(n_components=pca_components, random_state=42)

svm_model = make_pipeline(
    StandardScaler(),
    pca,
    SVC(kernel='linear', C=1.0, random_state=42)
)

print("\nTraining efficient SVM model (with PCA + linear kernel)...")
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nEfficient SVM Accuracy: {acc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
joblib.dump(svm_model, model_save_path)
print(f"\nModel saved to {model_save_path}")