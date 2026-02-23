import joblib
import numpy as np

from preprocess_Layer import clean_text
from risk_Layer import rule_based_risk_score
from logRegression import CustomLogisticRegression


# Load vectorizer 
vectorizer = joblib.load("../models/vectorizer.pkl")

# Load custom model parameters 
params = np.load("../models/Custom_LogisticRegression_params.npz")
w = params["w"]
b = float(params["b"])

# Recreate model and inject learned parameters
model = CustomLogisticRegression()
model.w = w
model.b = b


def predict(text: str) -> dict:

    # Clean text 

    cleaned = clean_text(text)

    # Vectorize 
    vec_sparse = vectorizer.transform([cleaned])

    # 3) Convert to dense for your custom model
    vec = vec_sparse.toarray().astype(np.float64)

    # 4) Custom model probability that class=1 (Scam)
    ai_prob = float(model.predict_proba(vec)[0][1])

    # 5) Rule-based score 
    rule_prob, rule_reasons= rule_based_risk_score(text)

    # 6) Combine scores
    final_score = float((ai_prob * 0.7) + (rule_prob * 0.3))

    return {
        "classification": "Scam" if final_score >= 0.5 else "Safe",
        "risk_score": round(final_score, 2),
        "ai_score": round(ai_prob, 2),
        "rule_score": round(rule_prob, 2),
        "reasons": rule_reasons
    }


if __name__ == "__main__":
    sample = "URGENT: verify your account now at http://fake-link.com"
    print(predict(sample))