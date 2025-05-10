
import streamlit as st
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Load and preprocess data
df = pd.read_csv("Project file.csv")
df = df[['class', 'message']].dropna()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

df['cleaned_message'] = df['message'].apply(preprocess_text)

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_message'])
y = df['class']

model = MultinomialNB()
model.fit(X, y)

# Streamlit UI
st.title("📩 Spam Detection App")
st.write("Enter a message below to classify it as **Spam** or **Valid**.")

user_input = st.text_area("Enter your message:", height=150)

if st.button("Classify"):
    cleaned = preprocess_text(user_input)
    vect_msg = vectorizer.transform([cleaned])
    prediction = model.predict(vect_msg)[0]
    st.subheader("🔍 Prediction:")
    st.success(f"This message is classified as **{prediction.upper()}**.")
