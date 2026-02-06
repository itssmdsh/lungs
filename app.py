import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
from PIL import Image
import gdown
import requests
from io import BytesIO
import os
import time

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Doctor Assistant Pro",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        text-align: center;
        margin-bottom: 20px;
        border-left: 6px solid #007bff;
    }
    .pred-title { font-size: 32px; font-weight: 800; margin-bottom: 5px; }
    .confidence-score { font-size: 20px; color: #555; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. GRAD-CAM ENGINE (The X-Ray Vision)
# ==========================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def find_target_layer(model):
    # Auto-detect the last convolutional layer
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer.name
    raise ValueError("Could not find a 4D layer.")

# ==========================================
# 3. MEDICAL KNOWLEDGE BASE
# ==========================================
DISEASE_INFO = {
    "Bacterial Pneumonia": {
        "symptoms": ["High fever & chills", "Productive cough (yellow/green mucus)", "Stabbing chest pain"],
        "care": ["Antibiotics", "Oxygen therapy", "Hydration"],
        "next_steps": "Blood culture & Sputum test recommended."
    },
    "Corona Virus Disease": {
        "symptoms": ["Fever & dry cough", "Loss of taste/smell", "Fatigue"],
        "care": ["Isolation", "Proning", "SpO2 Monitoring"],
        "next_steps": "RT-PCR Test & D-Dimer test."
    },
    "Tuberculosis": {
        "symptoms": ["Chronic cough (3+ weeks)", "Blood in sputum", "Night sweats"],
        "care": ["DOTS Therapy", "Isolation", "High-protein diet"],
        "next_steps": "Sputum smear microscopy & Chest CT."
    },
    "Normal": {
        "symptoms": ["No visible opacities", "Clear lung fields"],
        "care": ["Maintain healthy lifestyle", "Annual check-up"],
        "next_steps": "Routine follow-up."
    }
}

# ==========================================
# 4. MODEL LOADING
# ==========================================
@st.cache_resource
def load_system():
    model_path = 'Final_Hybrid_Model.keras'
    if not os.path.exists(model_path):
        with st.spinner("📥 Downloading AI Brain..."):
            file_id = '1OBkEkxsTK_V82RULwKgFQ10bvhV-xwnA'
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

try:
    model = load_system()
    CLASSES = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis']
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# ==========================================
# 5. DASHBOARD UI
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=80)
    st.title("Doctor Assistant")
    
    # --- INPUT METHOD SELECTION ---
    input_method = st.radio("Select Input Method:", ("📂 Upload File", "🌐 Image Link"))
    
    image_source = None
    
    if input_method == "📂 Upload File":
        uploaded_file = st.file_uploader("Upload X-Ray", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image_source = Image.open(uploaded_file).convert('RGB')
            
    elif input_method == "🌐 Image Link":
        url = st.text_input("Paste Image URL here:")
        st.caption("⚠️ **Note:** Link must be public and end in .jpg/.png")
        if url:
            try:
                response = requests.get(url)
                image_source = Image.open(BytesIO(response.content)).convert('RGB')
            except Exception as e:
                st.error("❌ Could not load image. Check the link.")

    st.markdown("---")
    st.write("### ⚙️ View Settings")
    show_gradcam = st.toggle("Show AI Attention (Heatmap)", value=False)
    alpha = st.slider("Heatmap Intensity", 0.0, 1.0, 0.4)

# ==========================================
# 6. MAIN LOGIC
# ==========================================
if image_source:
    col1, col2 = st.columns([1, 1.5], gap="medium")
    
    # Preprocessing
    img_array = np.array(image_source)
    original_size = image_source.size
    img_resized = tf.image.resize(img_array, (256, 256))
    img_batch = np.expand_dims(img_resized, axis=0)

    # Prediction
    preds = model.predict(img_batch)
    idx = np.argmax(preds[0])
    label = CLASSES[idx]
    conf = preds[0][idx] * 100
    
    with col1:
        st.write("#### 📷 Patient Scan")
        
        if show_gradcam:
            try:
                # Generate Heatmap
                target_layer = find_target_layer(model)
                heatmap = make_gradcam_heatmap(img_batch, model, target_layer)
                
                # Colorize Heatmap
                heatmap = np.uint8(255 * heatmap)
                jet = cm.get_cmap("jet")
                jet_colors = jet(np.arange(256))[:, :3]
                jet_heatmap = jet_colors[heatmap]
                
                # Resize Heatmap to original image size
                jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
                jet_heatmap = jet_heatmap.resize(original_size)
                jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
                
                # Superimpose
                superimposed_img = jet_heatmap * alpha + img_array
                superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
                
                st.image(superimposed_img, use_container_width=True, caption=f"AI Attention Map ({label})")
            except Exception as e:
                st.error(f"Grad-CAM Error: {e}")
                st.image(image_source, use_container_width=True)
        else:
            st.image(image_source, use_container_width=True, caption="Original Scan")

    with col2:
        st.write("#### 🔬 AI Diagnosis Report")
        
        # Color Logic
        color = "#28a745" if label == "Normal" else "#dc3545"
        icon = "✅" if label == "Normal" else "⚠️"
        
        # Result Card
        st.markdown(f"""
        <div class="metric-card" style="border-left: 8px solid {color};">
            <div class="pred-title" style="color: {color};">{icon} {label.upper()}</div>
            <div class="confidence-score">Certainty: <b>{conf:.2f}%</b></div>
            <p style="font-size:14px; margin-top:5px; color:#888;">Hybrid Neural Analysis</p>
        </div>
        """, unsafe_allow_html=True)

        # Confidence Breakdown
        with st.expander("📊 View Probability Breakdown", expanded=True):
            for i, class_name in enumerate(CLASSES):
                prob = preds[0][i]
                st.write(f"**{class_name}**: {prob*100:.2f}%")
                st.progress(float(prob))

        # Medical Insights
        info = DISEASE_INFO.get(label, {})
        t1, t2, t3 = st.tabs(["🤒 Symptoms", "💊 Treatment", "📋 Next Steps"])
        
        with t1:
            for s in info.get("symptoms", []): st.markdown(f"- {s}")
        with t2:
            for c in info.get("care", []): st.markdown(f"- {c}")
        with t3:
            st.info(info.get("next_steps"))

else:
    st.markdown("<br><br><h3 style='text-align:center; color:#999;'>waiting for input...</h3>", unsafe_allow_html=True)
