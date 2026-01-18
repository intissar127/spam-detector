# app/main.py

#pydantic :It makes sure your API receives correctly formatted data and rejects invalid data automatically.
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from training.config import MAX_LEN

app = FastAPI()

model = tf.keras.models.load_model("models/spam_lstm_model.keras")
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Define request model
class EmailRequest(BaseModel):
    email: str  # required field

@app.post("/predict")
def predict(request: EmailRequest):
    text = request.email
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    prob = float(model.predict(pad)[0][0])
    label = "spam" if prob > 0.5 else "ham"
    return {"label": label, "confidence": prob}
