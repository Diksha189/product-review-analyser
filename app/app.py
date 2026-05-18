from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__, template_folder="templates")

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
model_path = os.path.join(BASE_DIR, "..", "models", "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "..", "models", "vectorizer.pkl")

# Load model
model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

# Dataset path
data_path = os.path.join(BASE_DIR, "..", "data", "processed", "cleaned_balanced_dataset.csv")

# ✅ FIXED FUNCTION (returns BOTH result + probabilities)
def predict_sentiment(text):
    vec = vectorizer.transform([text])

    result = model.predict(vec)[0]

    # Check if model supports probability
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        classes = list(model.classes_)

        chart_data = [
            float(probs[classes.index("positive")]) if "positive" in classes else 0,
            float(probs[classes.index("negative")]) if "negative" in classes else 0,
            float(probs[classes.index("neutral")]) if "neutral" in classes else 0
        ]
    else:
        # fallback if probability not supported
        if result.lower() == "positive":
            chart_data = [0.7, 0.15, 0.15]
        elif result.lower() == "negative":
            chart_data = [0.15, 0.7, 0.15]
        else:
            chart_data = [0.2, 0.2, 0.6]

    return result, chart_data


# Home route
@app.route('/')
def home():
    try:
        df = pd.read_csv(data_path)
        df['sentiment'] = df['sentiment'].astype(str).str.lower().str.strip()

        dataset_chart = [
            len(df[df['sentiment'] == 'positive']),
            len(df[df['sentiment'] == 'negative']),
            len(df[df['sentiment'] == 'neutral'])
        ]
    except Exception as e:
        print("Dataset Error:", e)
        dataset_chart = None

    return render_template("index.html", dataset_chart=dataset_chart)


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']

    # ✅ get both result + chart
    result, chart_data = predict_sentiment(review)

    # emoji
    if result.lower() == "positive":
        emoji = "😊"
    elif result.lower() == "negative":
        emoji = "😡"
    else:
        emoji = "😐"

    # dataset chart
    try:
        df = pd.read_csv(data_path)
        df['sentiment'] = df['sentiment'].astype(str).str.lower().str.strip()

        dataset_chart = [
            len(df[df['sentiment'] == 'positive']),
            len(df[df['sentiment'] == 'negative']),
            len(df[df['sentiment'] == 'neutral'])
        ]
    except Exception as e:
        print("Dataset Error:", e)
        dataset_chart = None

    return render_template(
        "index.html",
        prediction=result,
        chart_data=chart_data,
        dataset_chart=dataset_chart,
        emoji=emoji
    )


if __name__ == "__main__":
    app.run(debug=True)