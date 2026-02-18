import re

def rule_based_risk_score(text: str) -> float:
    text = str(text)
    score = 0.0

    # URL signal
    if re.search(r"http[s]?://|www\.", text):
        score += 0.4

    # Phone signal (basic US patterns)
    if re.search(r"\b(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", text):
        score += 0.3

    # Email signal
    if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text):
        score += 0.3

    return min(score, 1.0)