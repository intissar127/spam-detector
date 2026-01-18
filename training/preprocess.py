# training/preprocess.py

import string
import re
from nltk.corpus import stopwords

def remove_punctuations(text):
    punctuations_list = string.punctuation
    temp = str.maketrans('', '', punctuations_list)
    return text.translate(temp)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = []
    for word in str(text).split():
        if word.lower() not in stop_words:
            words.append(word.lower())
    return " ".join(words)

def clean_text(text):
    text = text.replace("Subject", "")
    text = remove_punctuations(text)
    text = remove_stopwords(text)
    return text
