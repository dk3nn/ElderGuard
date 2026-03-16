import pickle

with open("scratch_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def predict_message(message):
    X = vectorizer.transform([message]).toarray()
    pred = model.predict(X, threshold=0.6)[0]
    prob = model.predict_proba(X)[0][1]

    return {
        "prediction": "scam" if pred == 1 else "safe",
        "confidence": float(prob),
        "explanation": "Classified using scratch logistic regression with TF-IDF features."
    }

if __name__ == "__main__":
    test_message = "Click this link to verify your bank account"
    print(predict_message(test_message))