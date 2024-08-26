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

# Setting page config
st.set_page_config(page_title="PDF Label Predictor", page_icon="📄", layout="wide")

# Custom CSS 
st.markdown("""
<style>
    body {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        background-attachment: fixed;
    }
    .stApp {
        max-width: 800px;
        margin: 0 auto;
        font-family: 'Arial', sans-serif;
        background-color: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        width: 50%; 
        background-color: #90EE90;
        color: white;
        border: none;
        padding: 10px 15px;
        border-radius: 5px;
        font-weight: bold;
        transition: background-color 0.3s;
        margin: 0 auto; 
        display: block;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>div>input {
        background-color: #f0f0f0;
        border: 1px solid #ddd;
        border-radius: 5px;
    }
    .stSelectbox>div>div>div {
        background-color: #f0f0f0;
        border: 1px solid #ddd;
        border-radius: 5px;
    }
    h1 {
        color: #2C3E50;
        text-align: center;
        margin-bottom: 30px;
    }
    .stAlert {
        padding: 10px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .prediction-result {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .prediction-label {
        font-weight: bold;
        color: #2C3E50;
    }
</style>
""", unsafe_allow_html=True)

# Caching functions
@st.cache_resource
def load_models():
    svm_model = joblib.load('models/svm_model.pkl')
    word2vec_model = Word2Vec.load("models/word2vec_model.bin")
    return svm_model, word2vec_model

@st.cache_resource
def load_ocr_model():
    os.environ['USE_TORCH'] = '1'
    return ocr_predictor(pretrained=True)

@st.cache_data
def load_label_encoder():
    with open("models/label_encoder.pkl", "rb") as f:
        label_encoder_array = pickle.load(f)
    unique_labels = np.unique(label_encoder_array)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = unique_labels
    return label_encoder

# Helper functions
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
    words = text.lower().split()
    vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]

    if len(vectors) > 0:
        flattened_vector = flatten_vectors(vectors, max_len=100)
    else:
        flattened_vector = np.zeros(10000)

    probabilities = svm_model.predict_proba([flattened_vector])[0]
    prediction_index = np.argmax(probabilities)
    prediction = label_encoder.inverse_transform([prediction_index])[0]
    probability = probabilities[prediction_index]

    return prediction, probability

# Main app
def main():
    st.title("📄 PDF Label Predictor")
    
    st.markdown("""
    <div style="background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
        <h4 style="color: #2C3E50; margin-bottom: 10px;">Welcome to the PDF Label Predictor!</h4>
        <p>Leverage cutting-edge machine learning to classify your PDF documents effortlessly. Just enter a PDF URL below and click 'Predict' to see the results!</p>
    </div>
    """, unsafe_allow_html=True)

    pdf_url = st.text_input("📎 Enter PDF URL:", key="pdf_url")
    model_choice = st.selectbox("🤖 Choose a model:", ["SVM"], key="model_choice")

    if st.button("🔍 Predict", key="predict_button"):
        if not pdf_url:
            st.error("❗ PDF URL is required")
        else:
            try:
                with st.spinner("⏳ Loading models..."):
                    svm_model, word2vec_model = load_models()
                    label_encoder = load_label_encoder()
                    ocr_model = load_ocr_model()

                with st.spinner("📄 Extracting text from PDF..."):
                    extracted_text = extract_text_from_pdf(pdf_url, ocr_model)
                
                if model_choice == 'SVM':
                    prediction, probability = predict_svm(extracted_text, svm_model, word2vec_model, label_encoder)
                    
                    # Display prediction and probability on the same line
                    st.markdown(f"""
                    <div class="prediction-result">
                        Prediction: <span class="prediction-label">{prediction}</span> (Probability: {probability:.2f})
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("❗ Invalid model choice")
            except Exception as e:
                st.error(f"❗ Error: {str(e)}")


if __name__ == "__main__":
    main()