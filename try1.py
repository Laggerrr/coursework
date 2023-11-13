import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import math
from tensorflow.keras.callbacks import EarlyStopping
import pickle
# Read the data from a CSV file
data = pd.read_csv("train.csv")

# Extract the text and author columns
texts = data['text']
authors = data['author']

# Encode the labels using LabelEncoder
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(authors)


# Split the data into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, encoded_labels, test_size=0.2, random_state=42)

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)

# Save the tokenizer


# Convert text to sequences
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# Pad sequences
len_train = [len(train_sequences[i]) for i in range(len(train_sequences))]
len_valid = [len(test_sequences [i]) for i in range(len(test_sequences ))]
len_ = np.array(len_train + len_valid)
maxlen = math.floor(len_.mean() + 2*len_.std()) + 1

train_sequences = pad_sequences(train_sequences, maxlen=maxlen)
test_sequences = pad_sequences(test_sequences, maxlen=maxlen)

# Convert labels to categorical format
num_classes = len(label_encoder.classes_)
train_labels = to_categorical(train_labels, num_classes=num_classes)
test_labels = to_categorical(test_labels, num_classes=num_classes)

# Load pre-trained word embeddings (e.g., GloVe)
embedding_dim = 100
embedding_index = {}
with open("glove.6B.100d.txt", encoding="utf8") as file:
    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embedding_index[word] = coefs


# Create an embedding matrix
vocab_size = len(tokenizer.word_index) + 1
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# Create the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=maxlen, weights=[embedding_matrix], trainable=True))
model.add(tf.keras.layers.SpatialDropout1D(0.5))
model.add(tf.keras.layers.LSTM(32))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))



early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(train_sequences, train_labels, validation_data=(test_sequences, test_labels), epochs=10, batch_size=32, callbacks=[early_stopping])

