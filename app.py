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
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Doctor Assistant Pro",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Medical CSS
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
    
    .pred-title {
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 5px;
    }
    
    .confidence-score {
        font-size: 18px;
        color: #555;
    }
    
    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. INTELLIGENT PREPROCESSING (AUTO-PILOT)
# ==========================================
def smart_preprocess_image(image_file):
    """
    Auto-detects inverted X-Rays (white background) and fixes them.
    Also handles standard resizing and RGB conversion.
    """
    img = image_file.convert('RGB')
    
    # Check corners for "White Background" (Inverted X-Ray)
    w, h = img.size
    corners = [
        img.getpixel((0, 0)), 
        img.getpixel((w-1, 0)), 
        img.getpixel((0, h-1)), 
        img.getpixel((w-1, h-1))
    ]
    avg_brightness = sum([sum(c)/3 for c in corners]) / 4
    
    # If corners are bright (>100), it's likely inverted or has a border
    if avg_brightness > 100:
        img = ImageOps.invert(img)
        # st.toast("⚠️ Auto-Corrected: Inverted colors for analysis", icon="🔧")
    
    # Resize to Model Standard
    img_array = np.array(img)
    img_resized = tf.image.resize(img_array, (256, 256))
    img_batch = np.expand_dims(img_resized, axis=0)
    
    return img_batch, img

# ==========================================
# 3. GRAD-CAM (X-RAY VISION)
# ==========================================
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
    # Search for the last 4D convolutional layer
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4: return layer.name
    raise ValueError("Could not find a target layer for Grad-CAM.")

# ==========================================
# 4. DATA & MODEL LOADING
# ==========================================
DISEASE_INFO = {
    "Bacterial Pneumonia": {
        "symptoms": ["High fever & chills", "Productive cough (yellow/green)", "Stabbing chest pain", "Fatigue"],
        "care": ["Antibiotics (Amoxicillin/Azithromycin)", "Oxygen therapy", "Fluid resuscitation"],
        "next_steps": "Urgent Pulmonology referral. Blood culture & Sputum test."
    },
    "Corona Virus Disease": {
        "symptoms": ["Fever", "Dry cough", "Loss of taste/smell", "Dyspnea (Shortness of breath)"],
        "care": ["Isolation (14 days)", "Antipyretics", "Proning (lying on stomach)", "SpO2 Monitoring"],
        "next_steps": "RT-PCR Confirmation. Monitor D-Dimer & CRP levels."
    },
    "Tuberculosis": {
        "symptoms": ["Chronic cough (3+ weeks)", "Hemoptysis (Blood in sputum)", "Night sweats", "Weight loss"],
        "care": ["DOTS Regimen (RIPE Therapy)", "Respiratory Isolation", "High-protein diet"],
        "next_steps": "Sputum AFB Smear & GeneXpert MTB/RIF test."
    },
    "Normal": {
        "symptoms": ["No abnormal opacities", "Clear costophrenic angles", "Normal cardiac silhouette"],
        "care": ["Maintain healthy lifestyle", "Annual check-up", "No treatment needed"],
        "next_steps": "Routine follow-up as per age protocols."
    }
}

@st.cache_resource
def load_system():
    model_path = 'Final_Hybrid_Model.keras'
    if not os.path.exists(model_path):
        with st.spinner("📥 Downloading AI Brain from Cloud..."):
            file_id = '1OBkEkxsTK_V82RULwKgFQ10bvhV-xwnA' # Your Drive ID
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

try:
    model = load_system()
    CLASSES = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis']
except Exception as e:
    st.error(f"Critical Error: {e}")
    st.stop()

# ==========================================
# 5. SIDEBAR & INPUT
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=70)
    st.title("Doctor Assistant")
    st.caption("v5.0 | Hybrid AI Diagnostic Tool")
    
    st.markdown("---")
    input_method = st.radio("Select Input Source:", ("📂 Upload File", "🌐 Image Link"))
    
    raw_image = None
    
    if input_method == "📂 Upload File":
        uploaded_file = st.file_uploader("Choose X-Ray", type=["jpg", "png", "jpeg"])
        if uploaded_file: raw_image = Image.open(uploaded_file)
            
    elif input_method == "🌐 Image Link":
        url = st.text_input("Paste Direct Image URL:")
        if url:
            try:
                # DISGUISE AS BROWSER TO FIX "INVALID LINK" ERRORS
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36'}
                response = requests.get(url, headers=headers, timeout=5)
                if response.status_code == 200:
                    raw_image = Image.open(BytesIO(response.content))
                else:
                    st.error(f"❌ Access Denied (Status {response.status_code})")
            except Exception as e:
                st.error("❌ Invalid Link. Ensure it ends in .jpg or .png")

    st.markdown("---")
    st.write("### ⚙️ Analysis Tools")
    show_gradcam = st.toggle("AI Vision (Heatmap)", value=False)
    alpha = st.slider("Heatmap Intensity", 0.0, 1.0, 0.4)

# ==========================================
# 6. MAIN DASHBOARD LOGIC
# ==========================================
if raw_image:
    # --- AUTO-PROCESSING ---
    img_batch, processed_image = smart_preprocess_image(raw_image)
    
    # --- INFERENCE ---
    preds = model.predict(img_batch)
    idx = np.argmax(preds[0])
    label = CLASSES[idx]
    conf = preds[0][idx] * 100
    
    # --- LAYOUT ---
    col1, col2 = st.columns([1, 1.5], gap="large")
    
    with col1:
        st.subheader("📷 Patient Scan")
        
        if show_gradcam:
            try:
                # Generate Heatmap
                target_layer = find_target_layer(model)
                heatmap = make_gradcam_heatmap(img_batch, model, target_layer)
                
                # Colorize
                heatmap = np.uint8(255 * heatmap)
                jet = cm.get_cmap("jet")
                jet_colors = jet(np.arange(256))[:, :3]
                jet_heatmap = jet_colors[heatmap]
                
                # Overlay
                jet_heatmap_img = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
                jet_heatmap_img = jet_heatmap_img.resize(processed_image.size)
                jet_heatmap_img = tf.keras.preprocessing.image.img_to_array(jet_heatmap_img)
                
                superimposed_img = jet_heatmap_img * alpha + np.array(processed_image)
                superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
                
                st.image(superimposed_img, use_container_width=True, caption=f"AI Attention: {label}")
            except Exception as e:
                st.error(f"Heatmap Error: {e}")
                st.image(processed_image, use_container_width=True)
        else:
            st.image(processed_image, use_container_width=True, caption="Auto-Enhanced Input")

    with col2:
        st.subheader("🔬 Diagnostic Report")
        
        # Dynamic Colors
        color = "#28a745" if label == "Normal" else "#dc3545"
        icon = "✅" if label == "Normal" else "⚠️"
        
        # 1. Main Result Card
        st.markdown(f"""
        <div class="metric-card" style="border-left: 8px solid {color};">
            <div class="pred-title" style="color: {color};">{icon} {label.upper()}</div>
            <div class="confidence-score">Certainty: <b>{conf:.2f}%</b></div>
            <p style="font-size:14px; margin-top:5px; color:#888;">Analysis by Titan Hybrid Model</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. Probability Breakdown
        with st.expander("📊 Full Probability Breakdown", expanded=True):
             for i, class_name in enumerate(CLASSES):
                prob = preds[0][i]
                # Highlight the predicted class
                txt_color = "red" if prob > 0.5 and class_name != "Normal" else "black"
                st.write(f"<span style='color:{txt_color}'>**{class_name}**: {prob*100:.2f}%</span>", unsafe_allow_html=True)
                st.progress(float(prob))
        
        # 3. Medical Insights (Tabs)
        info = DISEASE_INFO.get(label, {})
        t1, t2, t3 = st.tabs(["🤒 Symptoms", "💊 Recommended Care", "📋 Next Steps"])
        
        with t1:
            for s in info.get("symptoms", []): st.markdown(f"- {s}")
        with t2:
            for c in info.get("care", []): st.markdown(f"- {c}")
        with t3:
            st.info(f"**Action Plan:** {info.get('next_steps')}")
            if label != "Normal":
                st.warning("⚠️ Disclaimer: AI results must be verified by a certified Radiologist.")

else:
    # Empty State Hero
    st.markdown("""
    <br><br>
    <div style="text-align: center; color: #bdc3c7;">
        <h1>🩺 Waiting for Scan...</h1>
        <p>Please upload an X-Ray or paste a link to begin analysis.</p>
    </div>
    """, unsafe_allow_html=True)
