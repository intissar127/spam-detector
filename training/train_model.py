# training/train_model.py

import os
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd 
from config import *
from data_prep import load_data
from preprocess import clean_text


def main():
    # Load
    df = load_data(DATA_PATH)

    # Clean
    df["text"] = df["text"].apply(clean_text)

    # Balance
    ham = df[df["label"] == "ham"]
    spam = df[df["label"] == "spam"]
    ham_bal = ham.sample(n=len(spam), random_state=RANDOM_STATE)
    df_bal = pd.concat([ham_bal, spam]).reset_index(drop=True)

    # Encode labels
    df_bal["label"] = (df_bal["label"] == "spam").astype(int)

    # Split
    train_X, test_X, train_Y, test_Y = train_test_split(
        df_bal["text"], df_bal["label"],
        test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Tokenize
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_X)

    train_seq = tokenizer.texts_to_sequences(train_X)
    test_seq = tokenizer.texts_to_sequences(test_X)

    train_pad = pad_sequences(train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    test_pad = pad_sequences(test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

    # Model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(MAX_WORDS, EMBED_DIM, input_length=MAX_LEN),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_UNITS)),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    es = EarlyStopping(patience=3, monitor="val_accuracy", restore_best_weights=True)
    lr = ReduceLROnPlateau(patience=2, monitor="val_loss", factor=0.5)

    model.fit(
        train_pad, train_Y,
        validation_data=(test_pad, test_Y),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es, lr]
    )

    os.makedirs("models", exist_ok=True)
    model.save(MODEL_DIR)

    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    print("✅ Model and tokenizer saved!")


if __name__ == "__main__":
    main()
