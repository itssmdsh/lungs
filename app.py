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
    .pred-title { font-size: 30px; font-weight: 800; margin-bottom: 5px; }
    .confidence-score { font-size: 18px; color: #555; }
    /* Badge for unavailable features */
    .unavailable-badge {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
        font-size: 14px;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. INTELLIGENT PREPROCESSING
# ==========================================
def smart_preprocess_image(image_file):
    img = image_file.convert('RGB')
    
    # Auto-Invert Check
    w, h = img.size
    corners = [
        img.getpixel((0, 0)), 
        img.getpixel((w-1, 0)), 
        img.getpixel((0, h-1)), 
        img.getpixel((w-1, h-1))
    ]
    avg_brightness = sum([sum(c)/3 for c in corners]) / 4
    if avg_brightness > 100:
        img = ImageOps.invert(img)
    
    # Resize
    img_array = np.array(img)
    img_resized = tf.image.resize(img_array, (256, 256))
    img_batch = np.expand_dims(img_resized, axis=0)
    return img_batch, img

# ==========================================
# 3. ROBUST GRAD-CAM (THE FIX)
# ==========================================
def find_target_layer(model):
    """
    CRASH-PROOF SCANNER:
    Safely searches for the last 4D Convolutional layer.
    """
    # Search in reverse order (from output to input)
    for layer in reversed(model.layers):
        try:
            # 1. Check if layer has an output tensor
            # (Some layers like Dropout don't change shape, but we check anyway)
            output = layer.output
            
            # 2. Check dimensions. We need 4D: (Batch, Height, Width, Filters)
            if len(output.shape) == 4:
                # 3. Filter out Input layers
                if "input" not in layer.name.lower():
                    return layer.name
                    
        except AttributeError:
            # This catches the 'Dense object has no attribute output_shape' error
            # If a layer is weird, we just skip it and check the next one.
            continue
        except Exception:
            continue
            
    # If no 4D layer is found (e.g., pure Transformer or fully pooled model)
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    if last_conv_layer_name is None:
        return None

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

# ==========================================
# 4. DATA & MODEL LOADING
# ==========================================
DISEASE_INFO = {
    "Bacterial Pneumonia": {
        "symptoms": ["High fever", "Productive cough", "Chest pain"],
        "care": ["Antibiotics", "Oxygen therapy", "Hydration"],
        "next_steps": "Blood culture & Sputum test."
    },
    "Corona Virus Disease": {
        "symptoms": ["Fever", "Dry cough", "Loss of taste/smell"],
        "care": ["Isolation", "Antipyretics", "SpO2 Monitoring"],
        "next_steps": "RT-PCR Confirmation."
    },
    "Tuberculosis": {
        "symptoms": ["Chronic cough", "Blood in sputum", "Weight loss"],
        "care": ["DOTS Regimen", "Respiratory Isolation"],
        "next_steps": "Sputum AFB Smear & GeneXpert test."
    },
    "Normal": {
        "symptoms": ["No abnormalities"],
        "care": ["Maintain healthy lifestyle"],
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
    
    # Run the scanner ONCE at startup to see if Heatmaps are possible
    GRADCAM_LAYER = find_target_layer(model)
    GRADCAM_AVAILABLE = GRADCAM_LAYER is not None

except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# ==========================================
# 5. SIDEBAR & INPUT
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=70)
    st.title("Doctor Assistant")
    st.markdown("---")
    
    input_method = st.radio("Select Input:", ("📂 Upload File", "🌐 Image Link"))
    raw_image = None
    
    if input_method == "📂 Upload File":
        uploaded_file = st.file_uploader("Choose X-Ray", type=["jpg", "png", "jpeg"])
        if uploaded_file: raw_image = Image.open(uploaded_file)
            
    elif input_method == "🌐 Image Link":
        url = st.text_input("Paste Direct Image URL:")
        if url:
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=5)
                if response.status_code == 200:
                    raw_image = Image.open(BytesIO(response.content))
                else:
                    st.error(f"❌ Access Denied (Code {response.status_code}).")
            except Exception as e:
                st.error("❌ Invalid Link.")

    st.markdown("---")
    
    # Only show Grad-CAM controls if the model supports it
    if GRADCAM_AVAILABLE:
        st.write("### ⚙️ Visualization")
        show_gradcam = st.toggle("AI Vision (Heatmap)", value=False)
        alpha = st.slider("Heatmap Intensity", 0.0, 1.0, 0.4)
    else:
        st.markdown("""
        <div class="unavailable-badge">
            ⚠️ <b>Heatmap Unavailable</b><br>
            This hybrid model architecture hides internal layers, so AI Vision is disabled.
            Diagnostic accuracy is unaffected.
        </div>
        """, unsafe_allow_html=True)
        show_gradcam = False
        alpha = 0.4

# ==========================================
# 6. MAIN LOGIC
# ==========================================
if raw_image:
    img_batch, processed_image = smart_preprocess_image(raw_image)
    
    preds = model.predict(img_batch)
    idx = np.argmax(preds[0])
    label = CLASSES[idx]
    conf = preds[0][idx] * 100
    
    col1, col2 = st.columns([1, 1.5], gap="large")
    
    with col1:
        st.subheader("📷 Patient Scan")
        
        heatmap_generated = False
        if show_gradcam and GRADCAM_AVAILABLE:
            try:
                heatmap = make_gradcam_heatmap(img_batch, model, GRADCAM_LAYER)
                
                if heatmap is not None:
                    heatmap = np.uint8(255 * heatmap)
                    jet = cm.get_cmap("jet")
                    jet_colors = jet(np.arange(256))[:, :3]
                    jet_heatmap = jet_colors[heatmap]
                    
                    jet_heatmap_img = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
                    jet_heatmap_img = jet_heatmap_img.resize(processed_image.size)
                    jet_heatmap_img = tf.keras.preprocessing.image.img_to_array(jet_heatmap_img)
                    
                    superimposed_img = jet_heatmap_img * alpha + np.array(processed_image)
                    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
                    
                    st.image(superimposed_img, use_container_width=True, caption=f"AI Attention: {label}")
                    heatmap_generated = True
            except Exception as e:
                # If it fails during generation, fallback silently
                pass
        
        if not heatmap_generated:
             st.image(processed_image, use_container_width=True, caption="Analyzed Input")

    with col2:
        st.subheader("🔬 Diagnostic Report")
        color = "#28a745" if label == "Normal" else "#dc3545"
        icon = "✅" if label == "Normal" else "⚠️"
        
        st.markdown(f"""
        <div class="metric-card" style="border-left: 8px solid {color};">
            <div class="pred-title" style="color: {color};">{icon} {label.upper()}</div>
            <div class="confidence-score">Certainty: <b>{conf:.2f}%</b></div>
            <p style="font-size:14px; margin-top:5px; color:#888;">Hybrid Neural Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📊 Full Probability Breakdown", expanded=True):
             for i, class_name in enumerate(CLASSES):
                prob = preds[0][i]
                txt_color = "red" if prob > 0.5 and class_name != "Normal" else "black"
                st.write(f"<span style='color:{txt_color}'>**{class_name}**: {prob*100:.2f}%</span>", unsafe_allow_html=True)
                st.progress(float(prob))
        
        info = DISEASE_INFO.get(label, {})
        t1, t2, t3 = st.tabs(["🤒 Symptoms", "💊 Care", "📋 Next Steps"])
        with t1:
            for s in info.get("symptoms", []): st.markdown(f"- {s}")
        with t2:
            for c in info.get("care", []): st.markdown(f"- {c}")
        with t3:
            st.info(f"**Action Plan:** {info.get('next_steps')}")

else:
    st.markdown("<br><br><h3 style='text-align:center; color:#999;'>Waiting for input...</h3>", unsafe_allow_html=True)
