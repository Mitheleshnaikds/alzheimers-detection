import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import joblib

# Page configuration
st.set_page_config(
    page_title="Alzheimer's Disease Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.markdown("### 📤 Upload MRI Image")
st.sidebar.markdown("Upload a brain MRI scan for analysis")

# Select model architecture
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧩 Model Architecture")
selected_arch = st.sidebar.selectbox(
    label="Choose model",
    options=["cnn", "cnn_svm", "vgg16"],
    index=0,
    help="Pick which trained model to use"
)

# Prepare paths and placeholders
base_dir = os.path.dirname(__file__)
model = None
classifier = None
meta = None
feature_model = None
preprocess_fn = None

IMG_SIZE = (128, 128)

if selected_arch == 'cnn':
    model_path = os.path.join(base_dir, 'model.h5')
    IMG_SIZE = (128, 128)
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        st.sidebar.warning("Keras CNN model not found (model.h5).")
        st.sidebar.info("Train the CNN using train_model.py or place model.h5 in App/")

elif selected_arch == 'vgg16':
    # VGG16 end-to-end Keras model (expects 224x224 and ImageNet preprocessing)
    IMG_SIZE = (224, 224)
    vgg_path = os.path.join(base_dir, 'vgg16_alzheimer_model.h5')
    preprocess_fn = None
    if os.path.exists(vgg_path):
        try:
            model = load_model(vgg_path)
            preprocess_fn = tf.keras.applications.vgg16.preprocess_input
            st.sidebar.success(f'Loaded VGG16 model: {os.path.basename(vgg_path)}')
        except Exception as e:
            st.sidebar.error(f'Failed to load {os.path.basename(vgg_path)}: {e}')
    else:
        st.sidebar.warning('VGG16 model not found (vgg16_alzheimer_model.h5).')
        st.sidebar.info('Place vgg16_alzheimer_model.h5 in App/ to enable VGG16 predictions.')

else:  # cnn_svm
    IMG_SIZE = (128, 128)
    # Prefer a packaged end-to-end Keras hybrid model if present
    keras_hybrid_path = os.path.join(base_dir, 'model_cnn_svm.h5')
    if os.path.exists(keras_hybrid_path):
        try:
            model = load_model(keras_hybrid_path)
            st.sidebar.success('Loaded end-to-end Keras hybrid model (model_cnn_svm.h5).')
        except Exception as e:
            st.sidebar.error(f'Failed to load {os.path.basename(keras_hybrid_path)}: {e}')
    else:
        # Fallback: look for an sklearn classifier + a feature extractor model
        clf_path = os.path.join(base_dir, 'svm_model.pkl')
        feat_path = os.path.join(base_dir, 'cnn_feature_extractor.h5')
        # If user provided a feature-extractor model separately, prefer that
        if os.path.exists(clf_path):
            try:
                classifier = joblib.load(clf_path)
                st.sidebar.success(f'Loaded classifier: {os.path.basename(clf_path)}')
            except Exception as e:
                st.sidebar.error(f'Failed to load classifier {os.path.basename(clf_path)}: {e}')
                classifier = None

        # Load feature extractor: explicit extractor file or fall back to cnn_model.h5 / model.h5
        if os.path.exists(feat_path):
            try:
                feature_model = load_model(feat_path)
                st.sidebar.success(f'Loaded feature extractor: {os.path.basename(feat_path)}')
            except Exception as e:
                st.sidebar.error(f'Failed to load feature extractor {os.path.basename(feat_path)}: {e}')
                feature_model = None
        else:
            # try named CNN models
            for candidate in ['cnn_model.h5', 'model.h5']:
                cand_path = os.path.join(base_dir, candidate)
                if os.path.exists(cand_path):
                    try:
                        base_model = load_model(cand_path)
                        # try to find a named feature layer first
                        try:
                            feature_model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('feat_dense').output)
                        except Exception:
                            # fallback: use the penultimate layer
                            feature_model = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
                        st.sidebar.success(f'Using {candidate} for feature extraction')
                        break
                    except Exception as e:
                        st.sidebar.warning(f'Could not use {candidate} as feature extractor: {e}')
            if feature_model is None:
                st.sidebar.warning('No feature extractor found. Place cnn_feature_extractor.h5, cnn_model.h5 or model.h5 in App/.')

        if classifier is None or feature_model is None:
            st.sidebar.info('Hybrid pipeline incomplete: ensure both classifier (svm_model.pkl) and feature extractor are present in App/.')

# Modern CSS styling
st.markdown(
    """
    <style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #0a1428 0%, #1a2f5a 100%);
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Title styling */
    .title {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(120deg, #ffffff 0%, #e0e7ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Subtitle */
    .subtitle {
        color: #e0e7ff;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
        line-height: 1.6;
    }
    
    /* Info cards */
    .info-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #1a2f5a 0%, #1e3a5f 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        color: white;
        text-align: center;
    }
    
    /* Prediction text */
    .prediction-label {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 1rem 0;
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Confidence score */
    .confidence-score {
        font-size: 1.5rem;
        font-weight: 500;
        color: #e0e7ff;
        margin: 0.5rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1428 0%, #1a2f5a 100%);
    }
    
    /* Upload section */
    .upload-section {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px dashed rgba(255, 255, 255, 0.3);
    }
    
    /* File uploader styling */
    [data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(135deg, rgba(26, 47, 90, 0.4) 0%, rgba(30, 58, 95, 0.4) 100%) !important;
        border: 2px dashed rgba(255, 255, 255, 0.6) !important;
        border-radius: 15px !important;
        padding: 2rem !important;
    }
    
    [data-testid="stFileUploaderDropzone"]:hover {
        background: linear-gradient(135deg, rgba(26, 47, 90, 0.6) 0%, rgba(30, 58, 95, 0.6) 100%) !important;
        border-color: rgba(255, 255, 255, 0.9) !important;
    }
    
    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: white !important;
    }
    
    [data-testid="stFileUploaderDropzoneInstructions"] svg {
        fill: white !important;
    }
    
    [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    [data-testid="stBaseButton-secondary"] {
        background: #1e90ff !important;
        color: white !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stBaseButton-secondary"]:hover {
        background: #f0f0f0 !important;
        transform: scale(1.05) !important;
    }
    
    /* Hide toolbar - targeted approach */
    [data-testid="stToolbar"] {
        display: none !important;
    }
    
    /* Style the header bar */
    .st-emotion-cache-40nadn {
        background: linear-gradient(135deg, #0a1428 0%, #1a2f5a 100%) !important;
        border: none !important;
    }
    
    /* Alternative: Keep toolbar but style it nicely */
    /* Uncomment below if you want to show the toolbar
    [data-testid="stToolbar"] {
        background: transparent !important;
    }
    
    [data-testid="stToolbar"] button {
        color: white !important;
    }
    
    [data-testid="stToolbar"] svg {
        fill: white !important;
    }
    
    [data-testid="stAppDeployButton"] button {
        background: rgba(255, 255, 255, 0.2) !important;
        color: white !important;
        border-radius: 20px !important;
        padding: 0.5rem 1rem !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }
    
    [data-testid="stMainMenu"] button {
        color: white !important;
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 50% !important;
    }
    */
    
    /* Stage info boxes */
    .stage-box {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1e90ff;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .stage-title {
        font-weight: 700;
        color: #1e90ff;
        font-size: 1.1rem;
        margin-bottom: 0.3rem;
    }
    
    .stage-desc {
        color: #4a5568;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #1e90ff 0%, #4da6ff 100%);
        color: white;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] .sidebar-content {
        color: white;
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: white !important;
    }
    
    [data-testid="stSidebar"] p {
        color: #e0e7ff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown("<h1 class='title'>🧠 Alzheimer's Disease Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-Powered Brain MRI Analysis for Early Detection • Utilizing Deep Learning for Medical Diagnosis</p>", unsafe_allow_html=True)

def preprocess_image(image):
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Resize to model input size
    img_pil = Image.fromarray(img_array.astype('uint8')) if len(img_array.shape) == 2 else Image.fromarray(img_array)
    img_pil = img_pil.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    img_array = np.array(img_pil)
    
    # Ensure 3-channel RGB
    if len(img_array.shape) == 2:
        rgb_image = np.repeat(img_array[:, :, np.newaxis], 3, axis=2)
    elif img_array.shape[2] == 1:
        rgb_image = np.repeat(img_array, 3, axis=2)
    else:
        rgb_image = img_array[:, :, :3]
    
    # Preprocessing:
    # - If a specialized preprocess function is set (for ImageNet backbones), use it
    # - Otherwise normalize to [0,1]
    global preprocess_fn
    if 'preprocess_fn' in globals() and preprocess_fn is not None:
        img_proc = rgb_image.astype('float32')
        img_proc = preprocess_fn(img_proc)
    else:
        img_proc = rgb_image.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_proc, axis=0)
    return img_array

def predict(image):
    img_array = preprocess_image(image)
    # Hybrid path
    if selected_arch == 'cnn_svm':
        # If an end-to-end Keras hybrid model was loaded, use it directly
        if model is not None:
            prediction = model.predict(img_array)
            predicted_idx = int(np.argmax(prediction, axis=1)[0])
            confidence = float(np.max(prediction) * 100)
            return predicted_idx, confidence, prediction[0]

        # Otherwise fallback to classical classifier + feature extractor
        if classifier is None or feature_model is None:
            return None, None, None
        feats = feature_model.predict(img_array)
        if hasattr(classifier, 'predict_proba'):
            probs = classifier.predict_proba(feats)
        else:
            try:
                decision = classifier.decision_function(feats)
                exp = np.exp(decision - np.max(decision, axis=1, keepdims=True))
                probs = exp / np.sum(exp, axis=1, keepdims=True)
            except Exception:
                return None, None, None
        predicted_idx = int(np.argmax(probs, axis=1)[0])
        confidence = float(np.max(probs) * 100)
        return predicted_idx, confidence, probs[0]
    # Pure Keras model path
    if model is None:
        return None, None, None
    prediction = model.predict(img_array)
    predicted_idx = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    return predicted_idx, confidence, prediction[0]

# File uploader in sidebar
uploaded_file = st.sidebar.file_uploader(
    label="Choose an MRI image",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a brain MRI scan in JPG, JPEG, or PNG format"
)

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown("""
This AI model can classify brain MRI scans into four categories:
- **Non Demented**: No signs of dementia
- **Very Mild Demented**: Early-stage symptoms
- **Mild Demented**: Moderate progression
- **Moderate Demented**: Advanced symptoms
""")

st.sidebar.markdown("---")

# Main content area
if uploaded_file is None:
    # Welcome screen with information
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class='info-card'>
            <h2 style='color: #1e90ff; margin-bottom: 1rem;'>🔬 How It Works</h2>
            <ol style='color: #4a5568; line-height: 2;'>
                <li><strong>Upload</strong> a brain MRI scan image</li>
                <li>Our AI model <strong>analyzes</strong> the image</li>
                <li>Get instant <strong>classification results</strong></li>
                <li>View <strong>confidence scores</strong> for diagnosis</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-card'>
            <h2 style='color: #1e90ff; margin-bottom: 1rem;'>⚡ Key Features</h2>
            <ul style='color: #4a5568; line-height: 2;'>
                <li>✅ Deep Learning powered analysis</li>
                <li>✅ Instant results in seconds</li>
                <li>✅ High accuracy classification</li>
                <li>✅ User-friendly interface</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-card'>
            <h2 style='color: #1e90ff; margin-bottom: 1rem;'>📊 Classification Stages</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='stage-box'>
            <div class='stage-title'>🟢 Non Demented</div>
            <div class='stage-desc'>No cognitive impairment detected. Normal brain function.</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='stage-box'>
            <div class='stage-title'>🟡 Very Mild Demented</div>
            <div class='stage-desc'>Earliest stage with minor memory issues. Early intervention beneficial.</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='stage-box'>
            <div class='stage-title'>🟠 Mild Demented</div>
            <div class='stage-desc'>Noticeable cognitive decline. Assistance may be needed for daily tasks.</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='stage-box'>
            <div class='stage-title'>🔴 Moderate Demented</div>
            <div class='stage-desc'>Significant impairment. Requires substantial support and care.</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-card' style='margin-top: 2rem; text-align: center;'>
        <h3 style='color: #1e90ff;'>👈 Upload an MRI scan from the sidebar to get started</h3>
    </div>
    """, unsafe_allow_html=True)

else:
    # Analysis screen
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown("### 📸 Uploaded MRI Scan")
        st.image(image, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        with st.spinner('🔍 Analyzing MRI scan...'):
            predicted_idx, confidence, all_predictions = predict(image)
            if predicted_idx is None:
                st.error("Selected model is not available. Please train it first.")
                st.stop()
        
        class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
        predicted_label = class_labels[predicted_idx]
        
        # Emoji mapping
        emoji_map = {
            'Non Demented': '🟢',
            'Very Mild Demented': '🟡',
            'Mild Demented': '🟠',
            'Moderate Demented': '🔴'
        }
        
        st.markdown(f"""
        <div class='result-card'>
            <h2 style='color: #e0e7ff; margin-bottom: 1rem;'>📋 Diagnosis Result</h2>
            <div class='prediction-label'>{emoji_map[predicted_label]} {predicted_label}</div>
            <div class='confidence-score'>Confidence: {confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence breakdown
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Confidence Breakdown")
        
        for i, label in enumerate(class_labels):
            conf_value = all_predictions[i] * 100
            st.markdown(f"**{emoji_map[label]} {label}**")
            st.progress(float(all_predictions[i]))
            st.markdown(f"<p style='color: #1e90ff; font-weight: 600;'>{conf_value:.2f}%</p>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Additional information
    st.markdown("""
    <div class='info-card' style='margin-top: 2rem;'>
        <h3 style='color: #1e90ff;'>⚠️ Medical Disclaimer</h3>
        <p style='color: #4a5568; line-height: 1.8;'>
            This tool is designed for educational and research purposes only. It should not be used as a 
            substitute for professional medical advice, diagnosis, or treatment. Always consult with 
            qualified healthcare professionals for medical decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

