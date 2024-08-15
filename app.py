import os
import gdown
import torch
import joblib
import logging
import streamlit as st
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from gensim.models import Word2Vec
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Cache the loading of models
@st.cache_resource
def load_models():
    svm_model = joblib.load('svm_model.pkl')
    word2vec_model = Word2Vec.load("word2vec_model.bin")
    return svm_model, word2vec_model

# Cache the OCR predictor setup
@st.cache_resource
def load_ocr_model():
    os.environ['USE_TORCH'] = '1'
    return ocr_predictor(pretrained=True)

@st.cache_data
def load_label_encoder():
    with open("label_encoder.pkl", "rb") as f:
        label_encoder_array = pickle.load(f)

    unique_labels = np.unique(label_encoder_array)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = unique_labels
    return label_encoder

def fix_url(url):
    return url

def extract_text_from_pdf(pdf_url, ocr_model):
    pdf_url = fix_url(pdf_url)
    pdf_path = gdown.download(pdf_url, 'temp.pdf', quiet=True)
    document = DocumentFile.from_pdf(pdf_path)
    result = ocr_model(document)
    json_response = result.export()

    values = []
    num_pages_to_process = min(2, len(json_response['pages']))
    for page_index in range(num_pages_to_process):
        page = json_response['pages'][page_index]
        for block in page['blocks']:
            for line in block['lines']:
                for word in line['words']:
                    values.append(word['value'])

    return " ".join(values)

def pad_vectors(vectors, max_len=100):
    if len(vectors) > max_len:
        vectors = vectors[:max_len]
    else:
        vectors += [np.zeros(vectors[0].shape)] * (max_len - len(vectors))
    return np.array(vectors)

def flatten_vectors(vectors, max_len=100):
    padded_vectors = pad_vectors(vectors, max_len)
    return padded_vectors.reshape(-1)

def predict_svm(text, svm_model, word2vec_model, label_encoder):
    # Converting text to vector using Word2Vec
    words = text.lower().split()
    vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]

    if len(vectors) > 0:
        flattened_vector = flatten_vectors(vectors, max_len=100)
    else:
        # Creating a zero vector if no words are found
        flattened_vector = np.zeros(10000)

    # Predicting using SVM
    probabilities = svm_model.predict_proba([flattened_vector])[0]
    prediction_index = np.argmax(probabilities)
    prediction = label_encoder.inverse_transform([prediction_index])[0]
    probability = probabilities[prediction_index]

    return prediction, probability

# Streamlit UI
st.title("PDF Label Predictor")

pdf_url = st.text_input("Enter PDF URL:")
model_choice = st.selectbox("Choose a model:", ["SVM"])

if st.button("Predict"):
    if not pdf_url:
        st.error("PDF URL is required")
    else:
        try:
            # Load the models and OCR model only when the Predict button is pressed
            with st.spinner("Loading models..."):
                svm_model, word2vec_model = load_models()
                label_encoder = load_label_encoder()
                ocr_model = load_ocr_model()

            with st.spinner("Extracting text from PDF..."):
                extracted_text = extract_text_from_pdf(pdf_url, ocr_model)
            
            if model_choice == 'SVM':
                prediction, probability = predict_svm(extracted_text, svm_model, word2vec_model, label_encoder)
                st.success(f"Prediction: {prediction} (Probability: {probability:.2f})")
            else:
                st.error("Invalid model choice")
        except Exception as e:
            st.error(f"Error: {str(e)}")
