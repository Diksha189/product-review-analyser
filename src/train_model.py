import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


DATA_PATH = "data/processed/cleaned_balanced_dataset.csv"
MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"


if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Run preprocessing first.")

df = pd.read_csv(DATA_PATH)



df.columns = df.columns.str.strip().str.lower()

print("\n📊 Dataset columns:", df.columns)


df = df.dropna(subset=['clean_text', 'sentiment'])  # remove NaN rows
df = df[df['clean_text'].astype(str).str.strip() != ""]  # remove empty strings

# =========================
# FEATURES & LABELS
# =========================
X = df['clean_text']
y = df['sentiment']

print("\n📊 Class distribution:")
print(y.value_counts())

# =========================
# TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# TF-IDF VECTORIZATION
# =========================
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english'
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# MODEL TRAINING
# =========================
model = LogisticRegression(
    max_iter=1000,
    solver='lbfgs'
)

model.fit(X_train_vec, y_train)

# =========================
# PREDICTIONS
# =========================
y_pred = model.predict(X_test_vec)

# =========================
# EVALUATION
# =========================
print("\n📊 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================
os.makedirs("models", exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅ Model training completed successfully!")
print("📦 Model saved at:", MODEL_PATH)
print("📦 Vectorizer saved at:", VECTORIZER_PATH)