from fastapi import FastAPI

app = FastAPI(
    title="FinGraph Intelligence API",
    description="Fraud detection using federated learning and privacy-preserving techniques",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "FinGraph Intelligence API is running!",
        "status": "success"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }