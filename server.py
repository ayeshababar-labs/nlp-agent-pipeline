import os
from flask import Flask, request, jsonify
from app.predict import predict_sentiment
from app.model import load_model
from app.keywords import load_keybert_model, extract_keywords
from app.classifier import load_classifier_model, classify_topic
from app.agent import run_agent

app = Flask(__name__)

model = load_model()  # load models once at startup
keybert_model = load_keybert_model()  
classifier_model = load_classifier_model()  

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    result = predict_sentiment(model, data["text"])
    return jsonify(result)

@app.route("/keywords", methods=["POST"])
def keywords():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    result = extract_keywords(keybert_model, data["text"])
    return jsonify(result)

@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    result = classify_topic(classifier_model, data["text"])
    return jsonify(result)

@app.route("/agent", methods=["POST"])
def agent():
    data = request.get_json()

    if not data or "question" not in data or "text" not in data:
        return jsonify({"Missing 'question' or 'text' field"}), 400
    result = run_agent(data["question"], data["text"])
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)