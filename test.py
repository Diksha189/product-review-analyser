import pickle

model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

print(model.predict(vectorizer.transform(["worst product ever"])))
print(model.predict(vectorizer.transform(["very good product"])))