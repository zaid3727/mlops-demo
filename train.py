from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import mlflow
import joblib
import os
import numpy as np

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train model
clf = RandomForestClassifier()
clf.fit(X, y)

# Set experiment
mlflow.set_experiment("Iris_Classifier_Exp")

# Define input example (shape must match model input)
input_example = np.array([X[0]])  # or: np.array([[5.1, 3.5, 1.4, 0.2]])

# Start MLflow logging
with mlflow.start_run():
    mlflow.sklearn.log_model(
        clf,
        "model",
        input_example=input_example
    )
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("accuracy", clf.score(X, y))

# Save model to local disk
joblib.dump(clf, "iris_model.pkl")
print("Model trained, logged, and saved.")

