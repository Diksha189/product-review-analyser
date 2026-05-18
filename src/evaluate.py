import pandas as pd
import pickle

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# PATHS
# =========================
DATA_PATH = "data/processed/cleaned_balanced_dataset.csv"
MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open(MODEL_PATH, "rb"))
vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

# normalize columns
df.columns = df.columns.str.strip().str.lower()

# safety cleaning
df = df.dropna(subset=['clean_text', 'sentiment'])
df = df[df['clean_text'].astype(str).str.strip() != ""]

# =========================
# FEATURES & LABELS
# =========================
X = df['clean_text']
y = df['sentiment']

# vectorize
X_vec = vectorizer.transform(X)

# =========================
# PREDICT
# =========================
y_pred = model.predict(X_vec)

# =========================
# EVALUATION
# =========================
print("\n📊 Accuracy:", accuracy_score(y, y_pred))

print("\n📊 Classification Report:\n")
print(classification_report(y, y_pred))

print("\n📊 Confusion Matrix:\n")
print(confusion_matrix(y, y_pred))