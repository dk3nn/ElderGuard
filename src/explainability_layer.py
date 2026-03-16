import numpy as np
import re

def explain_prediction(text, vec, model, vectorizer, risk_score):
    explanation = {}

    # ---------- Model-based ---------- #
    if hasattr(model, "coef_"):
        feature_names = vectorizer.get_feature_names_out()
        vec_dense = vec.toarray()[0]
        weights = model.coef_[0]

        contributions = weights * vec_dense

        top_indices = np.argsort(contributions)[-5:]
        explanation["top_phrases"] = [
            feature_names[i] for i in reversed(top_indices)
        ]
    else:
        explanation["top_phrases"] = []

    # ---------- Rule-based ---------- #
    flags = []

    if re.search(r"http[s]?://|www\.", text):
        flags.append("Contains a link")

    if re.search(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", text):
        flags.append("Contains a phone number")

    if re.search(r"\burgent|immediately|act now\b", text.lower()):
        flags.append("Uses urgency language")

    if re.search(r"\bpassword|ssn|bank|account\b", text.lower()):
        flags.append("Requests sensitive information")

    explanation["red_flags"] = flags

    # ---------- Advice ----------
    if risk_score >= 0.7:
        advice = [
            "Do not click links or call numbers in this message.",
            "Verify using official contact information.",
            "Ask a trusted person if unsure."
        ]
    elif risk_score >= 0.4:
        advice = [
            "Be cautious and verify the sender before responding."
        ]
    else:
        advice = ["No strong scam indicators detected."]

    explanation["what_to_do"] = advice

    return explanation