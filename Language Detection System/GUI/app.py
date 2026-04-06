import streamlit as st
import joblib

# Load model
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("tfidf.pkl")
    encoder = joblib.load("encoder.pkl")
    return model, vectorizer, encoder

model, vectorizer, encoder = load_model()

st.set_page_config(page_title="Language AI", layout="centered")

# 🎨 CLEAN MODERN CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #667eea, #764ba2);
}

.title {
    text-align: center;
    font-size: 36px;
    font-weight: 700;
    color: white;
}

.subtitle {
    text-align: center;
    color: #e0e0e0;
    margin-bottom: 30px;
}

.flag-box {
    text-align: center;
    font-size: 40px;
    margin-bottom: 20px;
}

.result-box {
    margin-top: 25px;
    padding: 18px;
    border-radius: 12px;
    background: rgba(255,255,255,0.15);
    color: white;
    text-align: center;
    font-size: 20px;
    font-weight: 600;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #ff7eb3, #ff758c);
    color: white;
    font-size: 18px;
    border-radius: 10px;
    padding: 10px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown('<div class="title">🌍 Language Identification System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Clean • Professional • Smart</div>', unsafe_allow_html=True)

# INPUT (optional rakha)
user_input = st.text_area("✍ Enter Your  text Here:", height=120)

if st.button("🚀 Identify Language"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        transformed = vectorizer.transform([user_input])
        prediction = model.predict(transformed)[0]
        language = encoder.inverse_transform([prediction])[0]

        st.markdown(f'<div class="result-box">Identified Language: <strong>{language}</strong></div>', unsafe_allow_html=True)
           

st.markdown("<center style='color:white;margin-top:20px;'>⚡Supports Multiple Languages </center>", unsafe_allow_html=True)