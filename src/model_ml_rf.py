import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

DATASET_PATH = r"c:\Users\ramsu\Desktop\VS Projects\Aerial Image Classification\Dataset\processed\aerial_data.npz"
MODEL_SAVE_PATH = r"c:\Users\ramsu\Desktop\VS Projects\Aerial Image Classification\models\random_forest_final.pkl"

print("Dataset path:", DATASET_PATH)
print("Model will be saved to:", MODEL_SAVE_PATH)

data = np.load(DATASET_PATH, allow_pickle=True)
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
class_names = data["class_names"]

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
print(f"Number of classes: {len(class_names)}")

if len(X_train.shape) > 2:
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95, random_state=42)),
    ("rf", RandomForestClassifier(random_state=42, n_jobs=-1))
])

param_dist = {
    "rf__n_estimators": randint(120, 300),
    "rf__max_depth": [10, 15, 20, 25, None],
    "rf__min_samples_split": randint(2, 6),
    "rf__min_samples_leaf": randint(1, 3),
    "rf__max_features": ["sqrt", "log2", None],
    "rf__bootstrap": [True, False]
}

print("\nRunning efficient Random Forest hyperparameter search (optimized)...\n")

search = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

best_model = search.best_estimator_
print("\nBest Parameters:", search.best_params_)

y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nFinal Random Forest Accuracy: {accuracy:.4f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=class_names))

joblib.dump(best_model, MODEL_SAVE_PATH)
print(f"\nFinal model saved to {MODEL_SAVE_PATH}")