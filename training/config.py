# training/config.py
#Because config.py is just a central place to store settings and hyperparameters for your project.

#Instead of hardcoding values everywhere (like MAX_LEN = 100 in multiple files), you put them once in config.py so every script can read them.
DATA_PATH = "data/Emails.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

MAX_WORDS = 5000
MAX_LEN = 100

EMBED_DIM = 64
LSTM_UNITS = 32
DROPOUT = 0.3

BATCH_SIZE = 32
EPOCHS = 20

MODEL_DIR = "models/spam_lstm_model.keras"
TOKENIZER_PATH = "models/tokenizer.pkl"

# from tensorflow.keras.models import load_model

# model = load_model("models/spam_lstm_model.keras")  # or .h5
# If you want to serve the model in TF Serving or TFLite, you should use:

# model.export("models/spam_lstm_model")