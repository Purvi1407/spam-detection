import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Spam Detection App", layout="centered")

# -------------------------------
# 🎨 Premium UI Styling
# -------------------------------
st.markdown("""
<style>
/* Black Background */
.stApp {
    background-color: #0e1117;
    color: white;
}

/* Title */
h1 {
    text-align: center;
    color: #ffffff;
}

/* Card style */
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.6);
    margin-bottom: 20px;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #ff512f, #dd2476);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
    font-size: 16px;
}

/* Text area */
textarea {
    border-radius: 10px !important;
    background-color: #2b2f36 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load dataset
# -------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    data = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
    return data

data = load_data()

# -------------------------------
# Preprocess
# -------------------------------
data['label_num'] = data.label.map({'ham': 0, 'spam': 1})

X = data['message']
y = data['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Models
# -------------------------------
@st.cache_resource
def train_models():
    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('model', MultinomialNB())
    ])

    lr_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('model', LogisticRegression(max_iter=1000))
    ])

    nb_pipeline.fit(X_train, y_train)
    lr_pipeline.fit(X_train, y_train)

    nb_pred = nb_pipeline.predict(X_test)
    lr_pred = lr_pipeline.predict(X_test)

    nb_acc = accuracy_score(y_test, nb_pred)
    lr_acc = accuracy_score(y_test, lr_pred)

    return nb_pipeline, lr_pipeline, nb_acc, lr_acc

nb_model, lr_model, nb_acc, lr_acc = train_models()

# Best model
best_model = lr_model if lr_acc > nb_acc else nb_model
best_name = "Logistic Regression" if lr_acc > nb_acc else "Naive Bayes"

# -------------------------------
# Session State (History)
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# UI Header
# -------------------------------
st.markdown("<h1 style='color:#ff4b4b;'>📩 Spam Detection App</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; font-size:18px;'>🚀 Instantly detect spam messages using AI — fast, smart, and reliable</p>",
    unsafe_allow_html=True
)

# -------------------------------
# Input Card
# -------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("✉️ Enter Message")
user_input = st.text_area("Type your message here...")

predict_clicked = st.button("🚀 Predict")

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Prediction Section
# -------------------------------
if predict_clicked:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        with st.spinner("🔍 Analyzing message..."):
            prediction = best_model.predict([user_input])[0]
            proba = best_model.predict_proba([user_input])[0]

        confidence = max(proba) * 100

        # Progress bar
        st.progress(int(confidence))

        if prediction == 1:
            st.error(f"🚫 Spam Message ({confidence:.2f}% confidence)")
        else:
            st.success(f"✅ Not Spam ({confidence:.2f}% confidence)")

        # Save history
        st.session_state.history.append((user_input, prediction, confidence))

# -------------------------------
# Model Performance
# -------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("Naive Bayes", f"{nb_acc:.2f}")

with col2:
    st.metric("Logistic Regression", f"{lr_acc:.2f}")

st.write(f"🏆 Best Model: {best_name}")

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Prediction History
# -------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("📜 Prediction History")

for msg, pred, conf in st.session_state.history[::-1]:
    label = "Spam" if pred == 1 else "Not Spam"
    st.write(f"**{msg[:40]}...** → {label} ({conf:.1f}%)")

st.markdown('</div>', unsafe_allow_html=True)