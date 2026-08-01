# 📩 Spam Detection Web App (Streamlit + ML)

An interactive Machine Learning web application that detects whether a message is **Spam or Not Spam** using Natural Language Processing (NLP). Built with **Streamlit**, it provides real-time predictions, model comparison, and prediction history.


# Live Demo
(https://spam-detection-h9xkf2tyns4zfhao3kyemf.streamlit.app/)
---

## 🚀 Live Features

- 📊 Dual ML Models:
  - Naive Bayes
  - Logistic Regression
- 🧠 NLP using TF-IDF Vectorization
- ⚡ Real-time spam prediction
- 📈 Confidence score with progress bar
- 📝 Prediction history tracking
- 🎨 Modern dark-themed UI with Streamlit

---

## 🛠️ Tech Stack

- Python 🐍
- Streamlit 🎈
- Pandas & NumPy
- Scikit-learn 🤖
- NLP (TF-IDF Vectorizer)

---

## 📂 Project Structure

```

spam-detection-app/
│
├── app.py              # Main Streamlit app
├── requirements.txt    # Dependencies
└── README.md           # Project documentation

````

---

## ⚙️ How It Works

1. Dataset is loaded (SMS Spam Collection dataset)
2. Text is converted into numerical features using **TF-IDF**
3. Two ML models are trained:
   - Naive Bayes
   - Logistic Regression
4. Best performing model is selected automatically
5. User enters a message → model predicts spam or not spam

---

## ▶️ Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/spam-detection-app.git

# Navigate to project folder
cd spam-detection-app

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
````

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
```

---

## 📊 Example

* Input: *"Congratulations! You won a free iPhone. Click now!"*

* Output: 🚫 Spam

* Input: *"Hey, are we meeting tomorrow?"*

* Output: ✅ Not Spam

---

## 🧠 Future Improvements

* Add deep learning (LSTM / BERT)
* Deploy on Streamlit Cloud / Render
* Support multi-language spam detection
* Add email spam detection support

---

## 👨‍💻 Author

Purvi Lakhotia

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!

```

