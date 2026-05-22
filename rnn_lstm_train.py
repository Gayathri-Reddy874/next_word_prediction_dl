# IMPORT LIBRARIES
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SimpleRNN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# SAMPLE TEXT DATA
text = """deep learning is powerful
deep learning is fun
machine learning is interesting
artificial intelligence is future"""


# TOKENIZATION
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1

# Create sequences
input_sequences = []

for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    
    for i in range(1, len(token_list)):
        n_gram = token_list[:i+1]
        input_sequences.append(n_gram)

# Padding
max_len = max(len(x) for x in input_sequences)

input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_len, padding='pre'))

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# RNN MODEL
rnn_model = Sequential()
rnn_model.add(Embedding(total_words, 10, input_length=max_len-1))
rnn_model.add(SimpleRNN(50))
rnn_model.add(Dense(total_words, activation='softmax'))

rnn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
rnn_model.fit(X, y, epochs=200, verbose=0)
rnn_model.summary()

# LSTM MODEL
lstm_model = Sequential()
lstm_model.add(Embedding(total_words, 10, input_length=max_len-1))
lstm_model.add(LSTM(100))
lstm_model.add(Dense(total_words, activation='softmax'))

lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.fit(X, y, epochs=200, verbose=0)
lstm_model.summary()


# SAVE MODELS
rnn_model.save("rnn_model.h5")
lstm_model.save("lstm_model.h5")

pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))
pickle.dump(max_len, open("max_len.pkl", "wb"))

print("Models saved successfully!")