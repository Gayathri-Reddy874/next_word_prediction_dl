# 🧠 Next Word Prediction using RNN & LSTM

An AI-powered Deep Learning project that predicts the **next word in a sentence** using **RNN (Recurrent Neural Network)** and **LSTM (Long Short-Term Memory)** models built with TensorFlow and deployed using Streamlit.

---

## 🚀 Project Overview

This project demonstrates how sequential deep learning models can understand text patterns and predict the next probable word based on user input.

The application provides:
- 🔹 RNN-based next word prediction
- 🔹 LSTM-based next word prediction
- 🔹 Interactive Streamlit web interface
- 🔹 Text tokenization and sequence padding
- 🔹 Model training and deployment pipeline

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Streamlit
- Pickle
- Deep Learning
- NLP (Natural Language Processing)

---

## 📂 Project Structure

```bash
📁 Next-Word-Prediction/
│── rnn_lstm_train.py      # Model training script
│── rnn_lstm_app.py        # Streamlit web application
│── rnn_model.h5           # Saved RNN model
│── lstm_model.h5          # Saved LSTM model
│── tokenizer.pkl          # Saved tokenizer
│── max_len.pkl            # Saved sequence length
│── requirements.txt       # Required libraries
│── README.md              # Project documentation
```

---

## ⚙️ Features

✅ Train both RNN and LSTM models  
✅ Predict next word from user input  
✅ Compare RNN vs LSTM predictions  
✅ Interactive and user-friendly UI  
✅ Lightweight NLP project for beginners  

---

## 🧠 How It Works

1. Text data is tokenized using Keras Tokenizer.
2. Sequences are generated from sentences.
3. Sequences are padded for equal input length.
4. RNN and LSTM models are trained on the dataset.
5. User enters a sentence in the Streamlit app.
6. The model predicts the most probable next word.

---

## ▶️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Gayathri-Reddy874/next_word_prediction_dl.git
cd Next-Word-Prediction
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train Models

```bash
python rnn_lstm_train.py
```

### 4️⃣ Run Streamlit App

```bash
streamlit run rnn_lstm_app.py
```

---

## 💻 Sample Input & Output

### Input:
```text
deep learning is
```

### Output:
```text
powerful
```

---

## 📸 Application Preview

- User enters a sentence
- Selects either RNN or LSTM model
- Application predicts the next word instantly

---

## 📈 Future Enhancements

- Add larger datasets for better predictions
- Integrate GRU and Transformer models
- Deploy on Streamlit Cloud / Hugging Face
- Improve UI and prediction accuracy
- Add top-5 word predictions

---

## 🎯 Learning Outcomes

Through this project, I learned:
- Sequence modeling using Deep Learning
- RNN vs LSTM architecture comparison
- NLP preprocessing techniques
- TensorFlow model training and saving
- Streamlit deployment basics

---

## 👩‍💻 Author

**Mallareddygari Gayathri**  
Artificial Intelligence & Machine Learning Engineer  
Python | Deep Learning | NLP | Data Science Enthusiast

---

## ⭐ GitHub

If you found this project useful, give it a ⭐ on GitHub!
