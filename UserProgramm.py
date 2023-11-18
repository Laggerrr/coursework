import tkinter as tk
from tkinter import messagebox
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
import pickle


tokenizer = Tokenizer()
label_encoder = LabelEncoder()
maxlen = 65


tokenizer_path = "tokenizer.pkl"
label_encoder_path = "label_encoder.pkl"
tokenizer.word_index = pickle.load(open(tokenizer_path, 'rb'))
label_encoder = pickle.load(open(label_encoder_path, 'rb'))


model_path = "LSTM_Author_Model.h5"
model = tf.keras.models.load_model(model_path)


def preprocess_input(text):
    sequences = tokenizer.texts_to_sequences([text])
    sequences = pad_sequences(sequences, maxlen=maxlen)
    return sequences


def predict_author():
    user_text = text_input.get("1.0", tk.END).strip()
    words = user_text.split()
    if len(words) < 5:
        messagebox.showerror("Error", "Please enter up to 5 words. The model tends to work better if more words are given")
    else:
        # Продовжити з передбаченням автора
        input_sequences = preprocess_input(user_text)
        predictions = model.predict(input_sequences)
        predicted_labels = label_encoder.inverse_transform([np.argmax(predictions)])
        author_names = {
            'EAP': 'Edgar Allan Poe',
            'HPL': 'HP Lovecraft',
            'MWS': 'Mary Wollstonecraft Shelley'
        }
        predicted_author = author_names.get(predicted_labels[0])
        predicted_author_label.config(text="Predicted author: " + predicted_author)




window = tk.Tk()
window.title("Author Prediction")
window.geometry("800x350")


text_input = tk.Text(window, height=15, width=70)
text_input.pack(pady=10)


predicted_author_label = tk.Label(window, text="Predicted author: ")
predicted_author_label.pack()

predict_button = tk.Button(window, text="Predict", command=predict_author)
predict_button.pack(pady=10)

window.mainloop()
