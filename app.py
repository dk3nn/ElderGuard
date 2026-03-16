from flask import Flask, request, jsonify
from flask_cors import CORS
from predict_Layer import predict_message

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "ElderGuard backend running ✅"

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    message = data.get("message", "")

    if not message:
        return jsonify({"error": "No message provided"}), 400

    result = predict_message(message)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)