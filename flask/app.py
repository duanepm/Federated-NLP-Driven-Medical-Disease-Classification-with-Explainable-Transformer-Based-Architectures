import os
import re
import torch
import shap
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
from flask import Flask, request, render_template
from transformers import BertForSequenceClassification, BertTokenizerFast

app = Flask(__name__)

tokenizer = BertTokenizerFast.from_pretrained("tokenizer")
global_label_encoder = joblib.load("label_encoder.pkl")

num_labels = len(global_label_encoder.classes_)
global_model = BertForSequenceClassification.from_pretrained("bert_base_cased_local", num_labels=num_labels)
global_model.load_state_dict(torch.load("global_model1.pt", map_location=torch.device("cpu")))
global_model.eval()
device = torch.device("cpu")

global_model.to(device)

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

os.makedirs("static", exist_ok=True)

def preprocess_text(text):
    """Preprocess extracted text (remove short words, punctuation, and strip spaces)."""
    if not isinstance(text, str) or not text.strip():  
        return None 
    text = re.sub(r"(?i)patient report|patient name:.*|date:.*|hospital name:.*", "", text)
    text = re.sub(r'\b\w{1,2}\b', '', text)  
    symptoms_match = re.search(r"(?i)symptoms:\s*(.*)", text, re.DOTALL)
    text = symptoms_match.group(1) if symptoms_match else text
    text = re.sub(r'[^\w\s]', '', text)  
    text = re.sub(r"Hospital Name:.*", "", text)
    text = re.sub(r"\n+", "\n", text).strip()
    text = text.replace('|', 'I')
    text = re.sub(r"[^\x00-\x7F]+", '', text)
    return text.strip() if text.strip() else None 

def predict_disease(text):
    """Predict disease label using BERT model."""
    if not text:
        return "Error: No valid text extracted from the image."

    tokenized_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    with torch.no_grad():
        output = global_model(**tokenized_text)

    pred_label = torch.argmax(output.logits, dim=1).cpu().numpy()[0]
    return global_label_encoder.inverse_transform([pred_label])[0]

def explain_prediction(model, tokenizer, text, label_encoder):
    """Generate SHAP explanation and return HTML path."""
    if isinstance(text, pd.Series):
        text = text.astype(str).tolist()
    elif isinstance(text, str):
        text = [text]

    if not isinstance(text, list) or not all(isinstance(t, str) for t in text):
        raise ValueError("Input must be a string or a list of strings.")

    id2label = {i: label for i, label in enumerate(label_encoder.classes_)}

    def model_wrapper(input_texts):
        """Wrap model prediction for SHAP."""
        if isinstance(input_texts, np.ndarray):
            input_texts = input_texts.tolist()
        elif isinstance(input_texts, str):
            input_texts = [input_texts]

        tokenized = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            logits = model(**tokenized).logits
        return logits.cpu().numpy()

    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(model_wrapper, masker)
    shap_values = explainer(text)

    shap_values.output_names = [id2label[i] for i in range(len(id2label))]

    shap_html = shap.text_plot(shap_values, display=False)
    html_path = "static/shap_plot.html"

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(shap_html)

    return html_path

@app.route('/')
def home():
    """Render home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload, extract text, predict disease, and generate SHAP explanation."""
    if 'image' not in request.files:
        return render_template('index.html', prediction="No file uploaded", shap_html=None)

    image = request.files['image']
    img = Image.open(image)

    extracted_text = pytesseract.image_to_string(img)
    cleaned_text = preprocess_text(extracted_text)

    if not cleaned_text:
        return render_template('index.html', prediction="No valid text found in the image.", shap_html=None)

    predicted_disease = predict_disease(cleaned_text)

    shap_html_path = explain_prediction(global_model, tokenizer, cleaned_text, global_label_encoder)

    return render_template('index.html', prediction=predicted_disease, shap_html=shap_html_path)

if __name__ == "__main__":
    app.run(debug=True)
