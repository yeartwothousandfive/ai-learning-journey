
import json
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


sentences = [
    "I am happy",
    "I love this",
    "This is wonderful",
    "I feel great",
    "I am sad",
    "I hate this",
    "This is terrible",
    "I feel awful",
]

labels = [
    "positive",
    "positive",
    "positive",
    "positive",
    "negative",
    "negative",
    "negative",
    "negative",
]


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)


model = MultinomialNB()
model.fit(X, labels)

print("Word AI is ready!")

while True:
    text = input("\nEnter a sentence (or 'quit'): ")

    if text.lower() == "quit":
        break

    
    X_new = vectorizer.transform([text])

    
    prediction = model.predict(X_new)[0]
    probabilities = model.predict_proba(X_new)[0]
    confidence = max(probabilities) * 100

    print("Prediction:", prediction)
    print("Model confidence:", round(confidence, 1), "%")
