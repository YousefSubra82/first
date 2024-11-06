
import streamlit as st
import re
import joblib
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# Download NLTK stopwords if not already present
nltk.download('stopwords')

# Define preprocessing functions
def remove_digits(text):
    pattern = r'[^a-zA-Z.,!?/:;\"\'\s]'
    return re.sub(pattern, '', text)

def remove_special_characters(text):
    pat = r'[^a-zA-Z0-9.,!?/:;\"\'\s]'
    return re.sub(pat, '', text)

def non_ascii(s):
    return "".join(i for i in s if ord(i) < 128)

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return ' '.join([word for word in text.split() if word not in stop_words])

def punct(text):
    token = RegexpTokenizer(r'\w+')
    text = token.tokenize(text)
    return " ".join(text)

def lower(text):
    return text.lower()

# Apply all preprocessing steps
def clean_text(text):
    text = remove_digits(text)
    text = remove_special_characters(text)
    text = remove_stop_words(text)
    text = non_ascii(text)
    text = punct(text)
    text = lower(text)
    return text

# Load resources
@st.cache_resource
def load_resources():
    try:
        model_path = r'C:\Users\CS\Downloads\graduation project\11Task 1 For NLP preprocessing DataSet\Task 1 For NLP preprocessing DataSet\svm_model.sav'
        encoder_path = r'C:\Users\CS\Downloads\graduation project\11Task 1 For NLP preprocessing DataSet\Task 1 For NLP preprocessing DataSet\label_encoder_classes.sav'
        tfidf_vectorizer_path = r'C:\Users\CS\Downloads\graduation project\11Task 1 For NLP preprocessing DataSet\Task 1 For NLP preprocessing DataSet\tfidf_vectorizer.sav'
        
        # Load model and objects
        loaded_model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)

        # Ensure encoder has the necessary method
        if not hasattr(encoder, 'inverse_transform'):
            st.error("Loaded encoder is invalid. Please verify the encoder file.")
            st.stop()

        return loaded_model, encoder, tfidf_vectorizer
    
    except Exception as e:
        st.error("Error loading model or vectorizer. Please check the file paths and ensure models are saved properly.")
        st.write(f"Detailed error: {e}")
        st.stop()

# Streamlit App Interface
st.title("Disease Symptom Classification")
st.write("Enter symptoms to predict the possible disease.")

# Load resources
loaded_model, encoder, tfidf_vectorizer = load_resources()

# User input for symptoms
user_input = st.text_area("Enter symptoms:")

if st.button("Classify Disease"):
    if user_input:
        preprocessed_text = clean_text(user_input)
        st.write("**Preprocessed Text:**")
        st.write(preprocessed_text)
        # Transform input text to TF-IDF
        tfidf_features = tfidf_vectorizer.transform([preprocessed_text]).toarray()

        # Predict disease
        prediction_encoded = loaded_model.predict(tfidf_features)[0]
        prediction_label = encoder.inverse_transform([prediction_encoded])[0]
        prediction_probabilities = loaded_model.predict_proba(tfidf_features)[0]
        st.subheader("Predicted Disease")
        st.write(f"The model predicts the symptoms relate to: **{prediction_label}**")
        st.write(f"Probability of prediction: **{max(prediction_probabilities) * 100:.2f}%**")

    else:
        st.write("Please enter symptoms to classify.")

