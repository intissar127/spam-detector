# app/main.py

#pydantic :It makes sure your API receives correctly formatted data and rejects invalid data automatically.
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from training.config import MAX_LEN
import logging 
import os
from prometheus_client import Counter,Histogram,generate_latest
from fastapi.responses import Response
import time
app = FastAPI()
REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["endpoint"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])


BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

model_path = os.path.join(BASE_PATH, "models", "spam_lstm_model.keras")
tokenizer_path = os.path.join(BASE_PATH, "models", "tokenizer.pkl")

# Chargement
model = tf.keras.models.load_model(model_path)
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

# Define request model
class EmailRequest(BaseModel):
    email: str  # required field
logging.basicConfig(
    filename="logs/api.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
@app.post("/predict")
def predict(request: EmailRequest):
    start_time = time.time()
    REQUEST_COUNT.labels(endpoint="/predict").inc()
    logging.info(f"Prediction request:{request.email}")
    text = request.email
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    prob = float(model.predict(pad)[0][0])
    label = "spam" if prob > 0.5 else "ham"
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time()-start_time)
    return {"label": label, "confidence": prob}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(),media_type="text/plain")