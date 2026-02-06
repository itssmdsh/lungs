import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import time

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Doctor Assistant",
    page_icon="🩺",
    layout="wide",  # Uses full screen width
    initial_sidebar_state="expanded"
)

# Custom CSS for "Medical Dashboard" Look
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Card Styling */
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 15px;
        text-align: center;
    }
    
    /* Header Styling */
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Prediction Label Styling */
    .pred-label {
        font-size: 28px;
        font-weight: bold;
        margin: 10px 0;
    }
    
    /* Success/Critical Colors */
    .normal { color: #27ae60; }
    .disease { color: #e74c3c; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADER (Auto-Download)
# ==========================================
@st.cache_resource
def load_system():
    model_path = 'Final_Hybrid_Model.keras'
    
    if not os.path.exists(model_path):
        with st.spinner("📥 Downloading AI Brain from Secure Cloud..."):
            file_id = '1OBkEkxsTK_V82RULwKgFQ10bvhV-xwnA'
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, model_path, quiet=False)
            
    model = tf.keras.models.load_model(model_path)
    return model

try:
    model = load_system()
    CLASSES = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis']
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# ==========================================
# 3. SIDEBAR (Control Panel)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=80)
    st.title("Doctor Assistant")
    st.caption("AI-Powered Radiologist Aid")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📂 Upload Patient X-Ray", type=["jpg", "png", "jpeg"])
    
    st.markdown("---")
    st.info("ℹ️ **Privacy Note:** Uploaded scans are processed locally in RAM and not saved.")

# ==========================================
# 4. MAIN DASHBOARD
# ==========================================
st.markdown("### 🏥 Diagnostic Dashboard")

if uploaded_file is None:
    # Hero Section (Empty State)
    st.markdown("""
    <div style="text-align: center; padding: 50px; color: #95a5a6;">
        <h2>Ready for Analysis</h2>
        <p>Please upload a Chest X-Ray (CXR) from the sidebar to begin.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Layout: Left = Image, Right = Results
    col1, col2 = st.columns([1, 1.5], gap="large")
    
    with col1:
        st.markdown("#### 📷 Patient Scan")
        image = Image.open(uploaded_file).convert('RGB')
        # Display with rounded corners
        st.image(image, use_container_width=True)
        
    with col2:
        st.markdown("#### 🔬 AI Analysis Report")
        
        # Loading Animation
        with st.spinner("Running Hybrid Model Analysis..."):
            time.sleep(1) # UX smoothing
            
            # Preprocessing
            img_array = np.array(image)
            img_resized = tf.image.resize(img_array, (256, 256))
            img_batch = np.expand_dims(img_resized, axis=0)
            
            # Prediction
            preds = model.predict(img_batch)
            idx = np.argmax(preds[0])
            label = CLASSES[idx]
            conf = preds[0][idx] * 100
            
            # Dynamic Styling
            status_class = "normal" if label == "Normal" else "disease"
            icon = "✅" if label == "Normal" else "⚠️"
            
            # --- MAIN RESULT CARD ---
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0; color:#7f8c8d;">Primary Diagnosis</h3>
                <div class="pred-label {status_class}">{icon} {label}</div>
                <p style="font-size: 18px;">Confidence: <b>{conf:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            # --- DETAILED PROBABILITIES ---
            st.markdown("##### Detailed Confidence Levels")
            for i, class_name in enumerate(CLASSES):
                prob = preds[0][i]
                st.write(f"**{class_name}**")
                st.progress(float(prob)) # Requires float between 0.0 and 1.0

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 12px; color: #95a5a6;">
    <b>Doctor Assistant AI v2.0</b> | Powered by Titan Hybrid Architecture<br>
    Disclaimer: This tool is for investigational use only and does not replace professional medical advice.
</div>
""", unsafe_allow_html=True)
