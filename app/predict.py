import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import string
from nltk.corpus import stopwords
import nltk

# ===== DOWNLOAD STOPWORDS =====
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
punctuation_list = string.punctuation

# ===== LOAD MODEL =====
try:
    model = tf.keras.models.load_model("models/spam_lstm_model.keras")
    model_type = "keras"
except FileNotFoundError:
    raise FileNotFoundError("Keras model not found at models/spam_lstm_model.keras")

# ===== LOAD TOKENIZER =====
try:
    tokenizer = joblib.load("models/tokenizer.pkl")
    max_len = 100  # must match training
except FileNotFoundError:
    raise FileNotFoundError("Tokenizer not found at models/tokenizer.pkl")

# ===== TEXT PREPROCESSING =====
def preprocess_text(text: str) -> str:
    # Remove punctuation
    text = text.translate(str.maketrans("", "", punctuation_list))
    # Remove stopwords
    text = " ".join([w.lower() for w in text.split() if w.lower() not in stop_words])
    return text

# ===== PREDICTION =====
def predict_email(text: str) -> dict:
    clean_text = preprocess_text(text)

    # Convert to sequences
    seq = tokenizer.texts_to_sequences([clean_text])
    seq_padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')

    # Predict
    pred_prob = float(model.predict(seq_padded)[0][0])
    label = "spam" if pred_prob >= 0.5 else "ham"

    return {"label": label, "confidence": pred_prob}
