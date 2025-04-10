from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load("iris_model.pkl")

# Initialize FastAPI
app = FastAPI()

# Define request body schema
class Features(BaseModel):
    features: list[float]  # expecting 4 floats

# Define prediction endpoint
@app.post("/predict")
def predict(input: Features):
    data = np.array(input.features).reshape(1, -1)
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}