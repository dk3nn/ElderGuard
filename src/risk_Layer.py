import re

def rule_based_risk_score(text: str):
    text_raw = str(text)
    text = text_raw.lower()
    score = 0.0
    reasons = []

    # Rule-based heuristics with weighted scoring
    if re.search(r"http[s]?://|www\.", text):
        score += 0.4
        reasons.append("Contains a link.")


    if re.search(r"\b(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", text):
        score += 0.3
        reasons.append("Contains a phone number.")


    if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text):
        score += 0.3
        reasons.append("Contains an email address.")


    if re.search(r"\burgent|immediately|act now|limited time|asap\b", text):
        score += 0.25
        reasons.append("Uses urgency language.")


    if re.search(r"\bpay|payment|transfer|wire|bitcoin|gift card|fee\b", text):
        score += 0.35
        reasons.append("Mentions payment methods.")


    if re.search(r"\baccount.*(locked|suspended|disabled)|verify.*account|security alert\b", text):
        score += 0.35
        reasons.append("Claims account/security problem.")

    
    if re.search(r"\birs|bank|police|government|social security|amazon\b", text):
        score += 0.2
        reasons.append("Impersonates an authority/organization.")

    score = min(score, 1.0)
    return score, reasons