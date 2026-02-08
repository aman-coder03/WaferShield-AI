import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image
import time
import os

st.set_page_config(
    page_title="WaferShield AI",
    page_icon="",
    layout="wide"
)

MODEL_PATH = "models/model_fp16.onnx"
CLASS_NAMES = [
    "Center",
    "Clean",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Random",
    "Scratch"
]

@st.cache_resource
def load_model():
    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )
    return session

session = load_model()
input_name = session.get_inputs()[0].name

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((320, 320))

    img_array = (np.array(image).astype(np.float32) / 255.0).astype(np.float16)

    img_array = np.transpose(img_array, (2, 0, 1))  # HWC → CHW
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dim

    return img_array

st.sidebar.title("System Information")

st.sidebar.markdown("### Model")
st.sidebar.markdown("- **Architecture:** EfficientNet-Lite0")
st.sidebar.markdown("- **Format:** FP16 ONNX")
st.sidebar.markdown("- **Size:** 6.76 MB")

st.sidebar.markdown("### Edge Benchmark")
st.sidebar.markdown("- **Accuracy:** 89.77% (ONNX)")
st.sidebar.markdown("- **Latency:** 8.55 ms")
st.sidebar.markdown("- **Throughput:** 116.9 img/sec")
st.sidebar.markdown("- **Provider:** ONNX Runtime CPU")

st.sidebar.markdown("---")
st.sidebar.markdown("Built for Edge AI Deployment")


st.title("WaferShield AI")
st.markdown("### Edge AI Semiconductor Defect Classification System")

st.markdown(
    """
    Upload a wafer inspection image to classify defect type using a lightweight 
    edge-optimized EfficientNet-Lite0 model deployed in FP16 ONNX format.
    """
)

st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload Wafer Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:

    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file)

    with col1:
        st.subheader("Original Wafer Image")
        st.image(image, width="stretch")

    # Preprocess
    input_tensor = preprocess_image(image)

    # Inference
    start_time = time.time()
    outputs = session.run(None, {input_name: input_tensor})
    end_time = time.time()

    inference_time_ms = (end_time - start_time) * 1000

    logits = outputs[0]
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    predicted_index = np.argmax(probabilities)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = probabilities[0][predicted_index] * 100

    with col2:
        st.subheader("Prediction Results")

        st.metric(
            label="Predicted Defect",
            value=predicted_class
        )

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.metric(
            label="Inference Time",
            value=f"{inference_time_ms:.2f} ms"
        )

        st.progress(float(confidence) / 100.0)

    st.markdown("---")

    st.subheader("Class Probability Distribution")

    prob_dict = {
        CLASS_NAMES[i]: float(probabilities[0][i])
        for i in range(len(CLASS_NAMES))
    }

    st.bar_chart(prob_dict)

    st.markdown("---")

    st.subheader("Explainability (Grad-CAM)")

    st.info(
        "Grad-CAM visualizations are generated during evaluation phase. "
        "You can integrate live Grad-CAM inference for production explainability."
    )

st.markdown("---")
st.markdown(
    """
    **WaferShield AI** – Lightweight Edge-AI system for real-time semiconductor defect inspection.  
    Designed for deployment in resource-constrained fabrication environments.
    """
)
