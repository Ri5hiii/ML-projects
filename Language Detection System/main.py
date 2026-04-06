import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

print("🚀 Starting...")

# ✅ LOAD CSV (FIXED)
print("📂 Loading dataset...")
df = pd.read_csv("dataset.csv")
print("✅ Dataset loaded")

# Rename columns if needed
df.columns = df.columns.str.strip()

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["Text"].apply(clean_text)

# Encode labels
le = LabelEncoder()
df["label"] = le.fit_transform(df["language"])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42
)

# Vectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Model
print("📊 Training model...")
model = LinearSVC()
model.fit(X_train_vec, y_train)

# Save
print("💾 Saving model...")
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "tfidf.pkl")
joblib.dump(le, "encoder.pkl")

print("✅ DONE!")