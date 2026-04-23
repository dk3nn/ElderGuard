from flask import Flask, request, jsonify
from flask_cors import CORS

from predict_Layer import predict
from explainability_layer import explain_prediction
from preprocess_Layer import clean_text
from risk_Layer import rule_based_risk_score
from logRegression import CustomLogisticRegression
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

vectorizer = joblib.load(os.path.join(BASE_DIR, "../models/vectorizer.pkl"))
params = np.load(os.path.join(BASE_DIR, "../models/Custom_LogisticRegression_params.npz"))

model = CustomLogisticRegression()
model.w = params["w"]
model.b = float(params["b"])


@app.route("/predict", methods=["POST"])
def run_predict():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    base_result = predict(text)

    cleaned = clean_text(text)
    vec_sparse = vectorizer.transform([cleaned])

    explanation = explain_prediction(
        text=text,
        vec=vec_sparse,
        model=model,
        vectorizer=vectorizer,
        risk_score=base_result["risk_score"]
    )

    result = {
        **base_result,
        "explanation": explanation
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)