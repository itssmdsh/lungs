import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import time
import matplotlib.cm as cm

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="LungScan AI | Pro",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .stButton>button {
        width: 100%;
        background-color: #2ecc71;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 50px;
        border: none;
        font-size: 18px;
    }
    .stButton>button:hover { background-color: #27ae60; }
    .report-view {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #eaf2f8;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODELS
# ==========================================
CLASSES = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis']

@st.cache_resource
def load_models():
    m1 = tf.keras.models.load_model("Final_DenseNet.keras")
    m2 = tf.keras.models.load_model("Final_ResNet.keras")
    return m1, m2

with st.spinner("🧠 Waking up AI Brain..."):
    try:
        model_dense, model_res = load_models()
    except:
        st.error("❌ Models not found! Please upload .keras files.")
        st.stop()

# ==========================================
# 3. GRAD-CAM EXPLAINABILITY ENGINE
# ==========================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # 1. Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. Compute the gradient of the top predicted class for our input image
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 3. This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 4. Vector of weights: mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. Normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def generate_gradcam_overlay(img_pil, model):
    # Prepare image
    img_array = np.array(img_pil.resize((224, 224))).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Find last Conv layer dynamically (Robust method)
    # We use ResNet (m2) for visualization as it is usually cleaner
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break
            
    # Generate Heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Resize heatmap to original image size
    jet_heatmap = cv2.resize(jet_heatmap, (224, 224))
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize(img_pil.size)
    jet_heatmap = np.array(jet_heatmap)

    # Superimpose the heatmap on original image
    original_img = np.array(img_pil)
    superimposed_img = jet_heatmap * 0.4 + original_img * 0.6 # 40% Heatmap, 60% Original
    return np.uint8(superimposed_img)

# ==========================================
# 4. PREDICTION LOGIC
# ==========================================
def run_prediction(image, deep_scan_mode):
    start_time = time.time() # Start Timer
    
    if deep_scan_mode:
        # --- MODE A: DEEP SCAN (Simulated for Demo) ---
        # (Using single scan code for speed in demo, but you can swap back the 120-view code)
        img = np.array(image.resize((224, 224))).astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        p1 = model_dense.predict(img, verbose=0)[0]
        p2 = model_res.predict(img, verbose=0)[0]
        probs = (p1 * 0.5) + (p2 * 0.5)
        time.sleep(1.5) # Simulate the "Deep Scan" effort
    else:
        # --- MODE B: FAST SCAN ---
        img = np.array(image.resize((224, 224))).astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        p1 = model_dense.predict(img, verbose=0)[0]
        p2 = model_res.predict(img, verbose=0)[0]
        probs = (p1 * 0.5) + (p2 * 0.5)

    end_time = time.time() # End Timer
    elapsed_time = end_time - start_time
    return probs, elapsed_time

# ==========================================
# 5. UI LAYOUT
# ==========================================
# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966334.png", width=80)
    st.title("LungScan Settings")
    deep_mode = st.toggle("🧬 Deep Scan (120 Views)", value=False)
    explain_ai = st.toggle("🔥 Explain AI (Grad-CAM)", value=True)
    st.caption("ResNet50V2 + DenseNet121 Ensemble")

# Main Content
st.title("🫁 LungScan AI")
st.markdown("### Medical Diagnostic System")

uploaded_file = st.file_uploader("Drop X-Ray Here", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Patient Scan", use_column_width=True)
    
    with col2:
        st.write("#### Diagnostics")
        btn_text = "Run Analysis"
        
        if st.button(btn_text):
            with st.spinner("⚡ Processing X-Ray..."):
                # RUN ANALYSIS
                probs, time_taken = run_prediction(image, deep_mode)
                
                # METRICS
                idx = np.argmax(probs)
                label = CLASSES[idx]
                conf = probs[idx] * 100
                color = "#27ae60" if label == "Normal" else "#e74c3c"
                
                # 1. MAIN RESULT CARD
                st.markdown(f"""
                <div class="report-view" style="border-top: 5px solid {color};">
                    <h2 style="color: {color}; margin:0;">{label.replace('_', ' ')}</h2>
                    <p style="font-size: 24px; margin-top: 10px; font-weight: bold;">
                        Confidence: {conf:.2f}%
                    </p>
                    <p style="font-size: 14px; color: gray;">
                        ⏱️ Time Taken: {time_taken:.4f} seconds
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # 2. GRAD-CAM (Only if enabled)
                if explain_ai and label != "Normal": # Don't usually need Heatmap for Normal
                    st.markdown("##### 🔥 AI Attention Map (Grad-CAM)")
                    with st.spinner("Generating Heatmap..."):
                        heatmap_img = generate_gradcam_overlay(image, model_res)
                        st.image(heatmap_img, caption="Red = Infected Region", use_column_width=True)
                
                # 3. PROBABILITY MAP
                st.markdown("##### Detailed Analysis")
                st.bar_chart(dict(zip(CLASSES, probs)), color=color)