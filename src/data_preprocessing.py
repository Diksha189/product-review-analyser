import pandas as pd
import re
import os
from sklearn.utils import resample
from nltk.corpus import stopwords
import nltk

# Download stopwords (safe to run multiple times)
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))



RAW_PATH = "data/raw/Dataset-SA.csv"
OUTPUT_PATH = "data/processed/cleaned_balanced_dataset.csv"



def clean_text(text):
    text = str(text).lower()

    # remove special characters, numbers, punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # tokenize
    words = text.split()

    # remove stopwords
    words = [w for w in words if w not in stop_words]

    return " ".join(words)



def preprocess_and_balance():

    # Load dataset
    df = pd.read_csv(RAW_PATH)

    print("\n📊 Original columns:")
    print(df.columns)

    print("\n📊 Original distribution:")
    print(df['Sentiment'].value_counts())

    # Keep only required columns
    df = df[['Review', 'Sentiment']].dropna()

    # Clean text
    df['clean_text'] = df['Review'].apply(clean_text)

    

    positive = df[df['Sentiment'] == "positive"]
    negative = df[df['Sentiment'] == "negative"]
    neutral = df[df['Sentiment'] == "neutral"]

    print("\n📊 Class sizes before balancing:")
    print("Positive:", len(positive))
    print("Negative:", len(negative))
    print("Neutral:", len(neutral))

    

    # Base target = smallest between positive and negative
    target = min(len(positive), len(negative))

    positive_balanced = resample(
        positive,
        replace=False,
        n_samples=target,
        random_state=42
    )

    negative_balanced = resample(
        negative,
        replace=False,
        n_samples=target,
        random_state=42
    )

    # Neutral is capped safely (no crash)
    neutral_target = min(len(neutral), target)

    neutral_balanced = resample(
        neutral,
        replace=False,
        n_samples=neutral_target,
        random_state=42
    )

    
    
    df_balanced = pd.concat([
        positive_balanced,
        negative_balanced,
        neutral_balanced
    ])

    # Shuffle dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42)

    
    os.makedirs("data/processed", exist_ok=True)
    df_balanced.to_csv(OUTPUT_PATH, index=False)

    print("\n📊 Final balanced distribution:")
    print(df_balanced['Sentiment'].value_counts())

    print("\n✅ Preprocessing completed successfully!")
    print("Saved at:", OUTPUT_PATH)


if __name__ == "__main__":
    preprocess_and_balance()