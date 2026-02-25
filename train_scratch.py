print("train_scratch.py started ✅")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from scratch_logistic import ScratchLogisticRegression

def main():
    print("Loading dataset...")

    # CHANGE THIS to your real CSV path
    DATA_PATH = "data/dataset.csv"

    df = pd.read_csv(DATA_PATH)
    print("Columns:", list(df.columns))
    print("Rows:", len(df))

    # CHANGE THESE to your real column names
    TEXT_COL = "text"
    LABEL_COL = "label"

    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        raise KeyError(
            f"Expected columns '{TEXT_COL}' and '{LABEL_COL}'. "
            f"Found: {list(df.columns)}"
        )

    texts = df[TEXT_COL].astype(str)
    labels = df[LABEL_COL].astype(int)

    print("Splitting train/test...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print("Vectorizing TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train = vectorizer.fit_transform(X_train_text).toarray()
    X_test = vectorizer.transform(X_test_text).toarray()

    print("Training scratch logistic regression...")
    model = ScratchLogisticRegression(lr=0.5, epochs=300, l2=0.1)
    model.fit(X_train, y_train.values)

    print("Predicting + evaluating...")
    preds = model.predict(X_test)

    tp = ((y_test.values == 1) & (preds == 1)).sum()
    tn = ((y_test.values == 0) & (preds == 0)).sum()
    fp = ((y_test.values == 0) & (preds == 1)).sum()
    fn = ((y_test.values == 1) & (preds == 0)).sum()

    acc = (tp + tn) / len(y_test)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    print("\n=== Scratch Logistic Regression Results ===")
    print("Accuracy :", round(acc, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1       :", round(f1, 4))
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print([[int(tn), int(fp)], [int(fn), int(tp)]])
    print("Done ✅")

if __name__ == "__main__":
    main()
