import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
from PIL import Image, ImageOps
import gdown
import requests
from io import BytesIO
import os

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Doctor Assistant Pro",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
# 2. INTERNAL BRAIN (Auto-Processing)
# ==========================================
def smart_preprocess_image(image_file):
    """
    Intelligent Preprocessing Pipeline:
    1. Converts to RGB (Standardizes channels)
    2. Detects if image is Inverted/White-Background (Internet junk)
    3. Auto-Corrects the inversion if needed
    4. Resizes to 256x256
    """
    # 1. Standardize to RGB
    img = image_file.convert('RGB')
    
    # 2. Auto-Invert Logic (Check Corners)
    # X-Rays should have BLACK backgrounds. If corners are WHITE, it's inverted.
    w, h = img.size
    # Sample 4 corners
    corners = [
        img.getpixel((0, 0)), 
        img.getpixel((w-1, 0)), 
        img.getpixel((0, h-1)), 
        img.getpixel((w-1, h-1))
    ]
    # Calculate average brightness of corners (0=Black, 255=White)
    avg_brightness = sum([sum(c)/3 for c in corners]) / 4
    
    # If corners are bright (>100), assume inverted/white background
    if avg_brightness > 100:
        img = ImageOps.invert(img)
        st.toast("⚠️ Detected White Background: Auto-Inverting image...", icon="🔧")

    # 3. Resize (Exact Training Dimensions)
    img_array = np.array(img)
    img_resized = tf.image.resize(img_array, (256, 256))
    
    # 4. Batch Dimension
    img_batch = np.expand_dims(img_resized, axis=0)
    
    return img_batch, img  # Return both tensor (for AI) and PIL image (for UI)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None: pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def find_target_layer(model):
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4: return layer.name
    raise ValueError("Could not find a 4D layer.")

# ==========================================
# 3. DATA & MODEL LOADING
# ==========================================
DISEASE_INFO = {
    "Bacterial Pneumonia": {
        "symptoms": ["High fever & chills", "Productive cough", "Stabbing chest pain"],
        "care": ["Antibiotics", "Oxygen therapy", "Hydration"],
        "next_steps": "Blood culture & Sputum test."
    },
    "Corona Virus Disease": {
        "symptoms": ["Fever", "Dry cough", "Loss of taste/smell", "Fatigue"],
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
# 4. DASHBOARD UI
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=80)
    st.title("Doctor Assistant")
    
    input_method = st.radio("Select Input:", ("📂 Upload File", "🌐 Image Link"))
    
    raw_image = None
    
    if input_method == "📂 Upload File":
        uploaded_file = st.file_uploader("Upload X-Ray", type=["jpg", "png", "jpeg"])
        if uploaded_file: raw_image = Image.open(uploaded_file)
            
    elif input_method == "🌐 Image Link":
        url = st.text_input("Paste Image URL:")
        if url:
            try:
                response = requests.get(url, timeout=5)
                raw_image = Image.open(BytesIO(response.content))
            except: st.error("❌ Invalid Link")

    st.markdown("---")
    st.write("### ⚙️ Visualization")
    show_gradcam = st.toggle("Show Heatmap", value=False)
    alpha = st.slider("Intensity", 0.0, 1.0, 0.4)

if raw_image:
    col1, col2 = st.columns([1, 1.5], gap="medium")
    
    # --- AUTO-PILOT PROCESSING ---
    # We call the smart function here
    img_batch, processed_image = smart_preprocess_image(raw_image)

    # Prediction
    preds = model.predict(img_batch)
    idx = np.argmax(preds[0])
    label = CLASSES[idx]
    conf = preds[0][idx] * 100
    
    with col1:
        st.write("#### 📷 Patient Scan")
        
        if show_gradcam:
            try:
                target_layer = find_target_layer(model)
                heatmap = make_gradcam_heatmap(img_batch, model, target_layer)
                
                # GradCAM Visualization Logic
                heatmap = np.uint8(255 * heatmap)
                jet = cm.get_cmap("jet")
                jet_colors = jet(np.arange(256))[:, :3]
                jet_heatmap = jet_colors[heatmap]
                jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
                jet_heatmap = jet_heatmap.resize(processed_image.size)
                jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
                
                # Overlay on the PROCESSED (Corrected) image
                superimposed_img = jet_heatmap * alpha + np.array(processed_image)
                superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
                
                st.image(superimposed_img, use_container_width=True, caption=f"AI Attention ({label})")
            except Exception as e:
                st.error(f"Heatmap Error: {e}")
        else:
            # Show the PROCESSED image so user sees if it was inverted
            st.image(processed_image, use_container_width=True, caption="Analyzed Input (Auto-Corrected)")

    with col2:
        st.write("#### 🔬 AI Diagnosis Report")
        
        color = "#28a745" if label == "Normal" else "#dc3545"
        icon = "✅" if label == "Normal" else "⚠️"
        
        st.markdown(f"""
        <div class="metric-card" style="border-left: 8px solid {color};">
            <div class="pred-title" style="color: {color};">{icon} {label.upper()}</div>
            <div class="confidence-score">Certainty: <b>{conf:.2f}%</b></div>
            <p style="font-size:14px; margin-top:5px; color:#888;">Hybrid Neural Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📊 Probability Breakdown", expanded=True):
             for i, class_name in enumerate(CLASSES):
                prob = preds[0][i]
                st.write(f"**{class_name}**: {prob*100:.2f}%")
                st.progress(float(prob))
        
        info = DISEASE_INFO.get(label, {})
        t1, t2, t3 = st.tabs(["Symptoms", "Treatment", "Next Steps"])
        with t1:
            for s in info.get("symptoms", []): st.markdown(f"- {s}")
        with t2:
            for c in info.get("care", []): st.markdown(f"- {c}")
        with t3:
            st.info(info.get("next_steps"))

else:
    st.markdown("<br><br><h3 style='text-align:center; color:#999;'>waiting for scan...</h3>", unsafe_allow_html=True)
