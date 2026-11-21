import torch
import joblib
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from federated_learning.model import AMLModel


# ==========================================
# 1. FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="FinGraph Intelligence API",
    description="Financial fraud detection using machine learning and federated learning",
    version="1.0.0"
)


# ==========================================
# 2. CORS CONFIGURATION
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 3. LOAD MODEL AND SCALER
# ==========================================

MODEL_PATH = "results/models/centralized_model.pth"
SCALER_PATH = "results/models/centralized_scaler.pkl"

model = AMLModel(input_size=9)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu")
)

model.eval()

scaler = joblib.load(SCALER_PATH)

print("Fraud detection model loaded successfully!")
print("Scaler loaded successfully!")


# ==========================================
# 4. TRANSACTION INPUT SCHEMA
# ==========================================

class Transaction(BaseModel):

    value: float
    time: int
    amount_log: float
    amount_deviation: float
    time_step: int
    sender_transaction_count: float
    receiver_transaction_count: float
    sender_total_amount: float
    receiver_total_amount: float


# ==========================================
# 5. HOME ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "message": "FinGraph Intelligence API is running!",
        "status": "success",
        "model": "Centralized Fraud Detection Model"
    }


# ==========================================
# 6. HEALTH CHECK
# ==========================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# ==========================================
# 7. FRAUD PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
def predict_fraud(transaction: Transaction):

    # Convert transaction into feature array
    features = np.array([[
        transaction.value,
        transaction.time,
        transaction.amount_log,
        transaction.amount_deviation,
        transaction.time_step,
        transaction.sender_transaction_count,
        transaction.receiver_transaction_count,
        transaction.sender_total_amount,
        transaction.receiver_total_amount
    ]], dtype=np.float32)

    # Apply saved scaler
    scaled_features = scaler.transform(features)

    # Convert features into PyTorch tensor
    input_tensor = torch.tensor(
        scaled_features,
        dtype=torch.float32
    )

    # Model prediction
    with torch.no_grad():
        probability = model(input_tensor).item()

    # Classification threshold
    threshold = 0.5

    prediction = 1 if probability >= threshold else 0

    # Determine classification and risk
    if prediction == 1:
        result = "FRAUD"
        risk_level = "HIGH"
    else:
        result = "LEGITIMATE"
        risk_level = "LOW"

    # Return API response
    return {
        "transaction": transaction.model_dump(),
        "fraud_probability": round(probability, 4),
        "prediction": prediction,
        "classification": result,
        "risk_level": risk_level,
        "threshold": threshold
    }