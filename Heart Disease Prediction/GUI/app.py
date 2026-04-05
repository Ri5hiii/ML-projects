# =========================================
# 1. IMPORTS
# =========================================
import streamlit as st
import numpy as np
import pickle

# =========================================
# 2. LOAD MODEL
# =========================================
model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# =========================================
# 3. PAGE CONFIG
# =========================================
st.set_page_config(page_title="Heart AI ❤️", layout="wide")

# =========================================
# 4. CUSTOM CSS (🔥 BEAUTIFUL UI)
# =========================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
    url("https://images.unsplash.com/photo-1588776814546-1ffcf47267a5");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Glass cards */
.card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    margin-bottom: 15px;
}

/* Title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #ff4b4b;
}

/* Result boxes */
.result-high {
    background: rgba(255, 0, 0, 0.7);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 26px;
    color: white;
}

.result-low {
    background: rgba(0, 255, 100, 0.7);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 26px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =========================================
# 5. LANGUAGE
# =========================================
lang = st.sidebar.radio("🌐 Language", ["English", "हिंदी"])

def t(en, hi):
    return en if lang == "English" else hi

# =========================================
# 6. TITLE
# =========================================
st.markdown(f'<div class="title">{t("❤️ Heart Disease Predictor","❤️ हृदय रोग भविष्यवाणी")}</div>', unsafe_allow_html=True)

# =========================================
# 7. INPUT UI (CARDS 🔥)
# =========================================
st.sidebar.header(t("Patient Details", "मरीज की जानकारी"))

age = st.sidebar.slider(t("Age","आयु"), 20, 100, 30, key="age")
sex = st.sidebar.selectbox(t("Sex","लिंग"), [0,1], key="sex")
cp = st.sidebar.slider(t("Chest Pain","छाती दर्द"), 0, 3, 1, key="cp")
trestbps = st.sidebar.slider(t("BP","रक्तचाप"), 80, 200, 120, key="trestbps")
chol = st.sidebar.slider(t("Cholesterol","कोलेस्ट्रॉल"), 100, 400, 200, key="chol")
fbs = st.sidebar.selectbox(t("Sugar >120","शुगर"), [0,1], key="fbs")
restecg = st.sidebar.slider(t("ECG","ईसीजी"), 0, 2, 1, key="restecg")
thalach = st.sidebar.slider(t("Heart Rate","हृदय गति"), 60, 220, 150, key="thalach")
exang = st.sidebar.selectbox(t("Exercise Pain","व्यायाम दर्द"), [0,1], key="exang")
oldpeak = st.sidebar.slider(t("Oldpeak","स्ट्रेस"), 0.0, 6.0, 1.0, key="oldpeak")
slope = st.sidebar.slider(t("Slope","स्लोप"), 0, 2, 1, key="slope")
ca = st.sidebar.slider(t("Vessels","रक्त वाहिकाएं"), 0, 4, 0, key="ca")
thal = st.sidebar.slider(t("Thal","थैलेसीमिया"), 0, 3, 1, key="thal")

input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                        restecg, thalach, exang, oldpeak,
                        slope, ca, thal]])

# =========================================
# 8. BUTTON
# =========================================
if st.button(t("🔍 Predict","🔍 भविष्यवाणी करें")):

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0][1]

    # RESULT
    if prediction[0] == 1:
        st.markdown(f'<div class="result-high">⚠️ {t("High Risk","उच्च जोखिम")}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-low">✅ {t("Low Risk","कम जोखिम")}</div>', unsafe_allow_html=True)

    # PROGRESS BAR
    st.subheader(t("Confidence","विश्वास स्तर"))
    st.progress(int(prob * 100))
    st.write(f"{round(prob*100,2)}%")

    # EXPLANATION
    st.subheader(t("🧠 Explanation","🧠 कारण"))

    reasons = []
    if age > 50: reasons.append(t("High Age","अधिक आयु"))
    if chol > 240: reasons.append(t("High Cholesterol","उच्च कोलेस्ट्रॉल"))
    if trestbps > 140: reasons.append(t("High BP","उच्च रक्तचाप"))
    if oldpeak > 2: reasons.append(t("Heart Stress","हृदय तनाव"))
    if exang == 1: reasons.append(t("Exercise Pain","व्यायाम दर्द"))

    if reasons:
        for r in reasons:
            st.write("•", r)
    else:
        st.write(t("All Normal","सब सामान्य"))

# =========================================
# 9. FOOTER
# =========================================
st.markdown("---")
st.write("🚀 Made by S.R.S. | ML Project")