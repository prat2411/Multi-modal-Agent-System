"""Small Flask API for product price predictions."""

import os

from flask import Flask, jsonify, request

from pricer.evaluator import Evaluator, GroqPredictor


app = Flask(__name__)


def build_prompt(title: str, description: str) -> str:
    description = description.strip() or "No description provided"
    return (
        "Estimate the price of this item:\n"
        f"Title: {title.strip()}\n"
        f"Description: {description}\n\n"
        "Price: $"
    )


def estimate_price(title: str, description: str) -> dict:
    """Return the raw model response and a formatted price."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")

    response = GroqPredictor(api_key=api_key)({"prompt": build_prompt(title, description)})
    price = Evaluator._parse_price(response)
    return {"response": response, "price": f"${price:,.2f}"}


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    title = str(data.get("title", "")).strip()
    description = str(data.get("description", "")).strip()

    if not title:
        return jsonify({"error": "title is required"}), 400

    try:
        return jsonify(estimate_price(title, description))
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 503
    except Exception as error:
        return jsonify({"error": f"Prediction failed: {error}"}), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))