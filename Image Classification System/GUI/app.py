import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

def set_bg():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1604187351574-c75ca79f5807");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()

st.markdown(
    """
    <style>
    .block-container {
        background-color: rgba(0, 0, 0, 0.6);
        padding: 2rem;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Garbage Classifier", page_icon="♻️", layout="centered")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

class_names = ['cardboard', 'paper', 'glass', 'trash', 'plastic', 'metal']

# =========================
# HEADER
# =========================
st.markdown(
    "<h1 style='text-align:center;color:green;'>♻️ Garbage Classification AI</h1>",
    unsafe_allow_html=True
)
st.markdown("<p style='text-align:center;'>Upload or capture image</p>", unsafe_allow_html=True)

st.markdown("---")

# =========================
# INPUT OPTIONS
# =========================
option = st.radio("Choose Input Method", ["Upload Image", "Use Camera"])

image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("📂 Upload Image", type=["jpg","png","jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)

else:
    camera_image = st.camera_input("📸 Take a picture")
    if camera_image:
        image = Image.open(camera_image)

# =========================
# PREPROCESS
# =========================
def preprocess(img):
    img = img.resize((224,224))
    img = np.array(img)/255.0
    return np.expand_dims(img, axis=0)

# =========================
# PREDICT
# =========================
def predict(img):
    processed = preprocess(img)
    pred = model.predict(processed)[0]
    return pred

# =========================
# DISPLAY RESULT
# =========================
if image:
    st.image(image, caption="📷 Input Image", use_column_width=True)

    pred = predict(image)

    idx = np.argmax(pred)
    label = class_names[idx]
    confidence = pred[idx] * 100

    st.markdown("### 🔍 Prediction Result")

    st.success(f"🗑️ **{label.upper()}**")
    st.metric("Confidence", f"{confidence:.2f} %")
    st.progress(int(confidence))

    # =========================
    # TOP 3 PREDICTIONS
    # =========================
    st.markdown("### 🔝 Top 3 Predictions")

    top3_idx = np.argsort(pred)[-3:][::-1]

    for i in top3_idx:
        st.write(f"{class_names[i]} : {pred[i]*100:.2f}%")

    # =========================
    # GRAPH (BAR CHART)
    # =========================
    st.markdown("### 📊 Prediction Probability Graph")

    fig = plt.figure()
    plt.bar(class_names, pred)
    plt.xlabel("Classes")
    plt.ylabel("Probability")
    plt.title("Prediction Distribution")

    st.pyplot(fig)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>🚀 Powered by MobileNetV2</p>",
    unsafe_allow_html=True
)