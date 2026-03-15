import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from preprocess_Layer import clean_text
from logRegression import CustomLogisticRegression  # your custom class


DATA_PATH = "../data/Dataset_10191.csv"

os.makedirs("../models", exist_ok=True)
os.makedirs("../results/plots", exist_ok=True)

# Load dataset

data = pd.read_csv(DATA_PATH)

# Convert labels to binary (ham=safe, spam/smishing=unsafe)

data["label"] = data["LABEL"].map({"ham": 0, "spam": 1, "smishing": 1})

# Clean message text

data["clean_text"] = data["TEXT"].astype(str).apply(clean_text)

# Removes any rows that didn't map cleanly
data = data.dropna(subset=["label", "clean_text"])
data["label"] = data["label"].astype(int)

# Vectorize text 
vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
X = vectorizer.fit_transform(data["clean_text"])
y = data["label"].to_numpy()

# Train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define sklearn models 
models = {
    "Baseline_MostFrequent": DummyClassifier(strategy="most_frequent"),
    "Sklearn_LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
    "LinearSVC": LinearSVC(),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        random_state=42,
        n_jobs=-1
    ),
}

results = []


# Train + evaluate sklearn models

for name, model in models.items():
    print("\n" + "=" * 70)
    print(f"MODEL: {name}")
    print("=" * 70)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1
    )

    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n",
          classification_report(y_test, y_pred, target_names=["Safe(0)", "Scam(1)"]))

    results.append({
        "model": name,
        "accuracy": acc,
        "precision_scam": precision,
        "recall_scam": recall,
        "f1_scam": f1,
        "tn": cm[0, 0],
        "fp": cm[0, 1],
        "fn": cm[1, 0],
        "tp": cm[1, 1],
    })

    # Save sklearn models 
    if name != "Baseline_MostFrequent":
        joblib.dump(model, f"../models/{name}.pkl")



# Train + evaluate CUSTOM Logistic Regression

print("\n" + "=" * 70)
print("MODEL: Custom_LogisticRegression")
print("=" * 70)

X_train_dense = X_train.toarray().astype(np.float64)
X_test_dense = X_test.toarray().astype(np.float64)


custom_model = CustomLogisticRegression(
    lr=0.02,
    epochs=3000,
    reg_strength=.01,
    verbose=True
)

custom_model.fit(X_train_dense, y_train)
probs = custom_model.predict_proba(X_test_dense)[:, 1]


best_t, best_f1 = 0.5, -1
best_stats = None

for t in np.arange(0.30, 0.71, 0.01):
    preds = (probs >= t).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary", pos_label=1, zero_division=0
    )
    if f1 > best_f1:
        best_f1 = f1
        best_t = float(t)
        best_stats = (precision, recall, f1, confusion_matrix(y_test, preds))

precision, recall, f1, cm = best_stats

y_pred_custom = (probs >= best_t).astype(int)


y_pred_custom = custom_model.predict(X_test_dense, threshold=0.5)

acc = accuracy_score(y_test, y_pred_custom)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred_custom, average="binary", pos_label=1
)

cm = confusion_matrix(y_test, y_pred_custom)

print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n",
      classification_report(y_test, y_pred_custom, target_names=["Safe(0)", "Scam(1)"]))

results.append({
    "model": "Custom_LogisticRegression",
    "accuracy": acc,
    "precision_scam": precision,
    "recall_scam": recall,
    "f1_scam": f1,
    "tn": cm[0, 0],
    "fp": cm[0, 1],
    "fn": cm[1, 0],
    "tp": cm[1, 1],
})

# Save custom LogReg model parameters 

np.savez("../models/Custom_LogisticRegression_params.npz", w=custom_model.w, b=custom_model.b)

# Save vectorizer 

joblib.dump(vectorizer, "../models/vectorizer.pkl")


# Save results table

results_df = pd.DataFrame(results).sort_values(by="f1_scam", ascending=False)
results_df.to_csv("../results/model_comparison.csv", index=False)
print("\nSaved: ../results/model_comparison.csv")
print(results_df)

# Create bar chart comparing metrics

plot_df = results_df.set_index("model")[["accuracy", "precision_scam", "recall_scam", "f1_scam"]]
ax = plot_df.plot(kind="bar", figsize=(10, 5))
ax.set_title("Model Comparison (Higher is Better)")
ax.set_ylabel("Score")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("../results/plots/model_comparison_bar.png")
plt.close()

print("Saved: ../results/plots/model_comparison_bar.png")

# Confusion matrixes for each model

def plot_cm(cm, title, path):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["Safe", "Scam"])
    plt.yticks([0, 1], ["Safe", "Scam"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

for row in results:
    cm = [[row["tn"], row["fp"]], [row["fn"], row["tp"]]]
    plot_cm(cm, f"Confusion Matrix: {row['model']}",
            f"../results/plots/confusion_{row['model']}.png")

print("Saved confusion matrices in: ../results/plots/")
print("Saved vectorizer in: ../models/vectorizer.pkl")
print("Saved custom params in: ../models/Custom_LogisticRegression_params.npz")
