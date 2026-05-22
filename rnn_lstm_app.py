import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# LOAD MODELS
rnn_model = load_model("rnn_model.h5")
lstm_model = load_model("lstm_model.h5")

tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
max_len = pickle.load(open("max_len.pkl", "rb"))


# STREAMLIT UI
st.title("🧠 Next Word Prediction (RNN & LSTM)")

input_text = st.text_input("Enter a sentence")

model_choice = st.selectbox("Choose Model", ["RNN", "LSTM"])

# PREDICTION FUNCTION
def predict_next_word(model, text):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')

    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted)

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

    return ""

# BUTTON
if st.button("Predict Next Word"):
    
    if model_choice == "RNN":
        word = predict_next_word(rnn_model, input_text)
    else:
        word = predict_next_word(lstm_model, input_text)

    st.success(f"Next Word: {word}")