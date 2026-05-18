import pickle
import os

# Load model and vectorizer
base_dir = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(base_dir, "models", "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(base_dir, "models", "vectorizer.pkl"), "rb"))

# Test data
test_reviews = [
    "This product is amazing",
    "Worst product ever",
    "It is okay"
]

# Run predictions
for r in test_reviews:
    vec = vectorizer.transform([r])
    print(r, "=>", model.predict(vec)[0])