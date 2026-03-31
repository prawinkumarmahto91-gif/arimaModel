import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# ── Load model once at startup ──────────────────────────────────────────────
MODEL_PATH = "arimaModel.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print(f"[INFO] Model loaded: {type(model)}")


# ── Health check ─────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": str(type(model).__name__)})


# ── Forecast endpoint ─────────────────────────────────────────────────────────
@app.route("/forecast", methods=["POST"])
def forecast():
    """
    POST /forecast
    Body (JSON):
    {
        "steps": 10          // number of future steps to forecast (required)
    }

    Response:
    {
        "forecast": [1.2, 3.4, ...],
        "steps": 10
    }
    """
    data = request.get_json(force=True)

    if not data or "steps" not in data:
        return jsonify({"error": "'steps' field is required in request body"}), 400

    steps = int(data["steps"])
    if steps <= 0:
        return jsonify({"error": "'steps' must be a positive integer"}), 400

    try:
        forecast_values = model.forecast(steps=steps)

        # Convert numpy types to plain Python for JSON serialization
        if hasattr(forecast_values, "tolist"):
            forecast_values = forecast_values.tolist()
        else:
            forecast_values = list(forecast_values)

        return jsonify({
            "steps": steps,
            "forecast": forecast_values
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Predict (in-sample) endpoint ──────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body (JSON):
    {
        "start": 0,     // start index for in-sample prediction (optional, default 0)
        "end": 20       // end index for in-sample prediction (required)
    }

    Response:
    {
        "predictions": [1.2, 3.4, ...],
        "start": 0,
        "end": 20
    }
    """
    data = request.get_json(force=True)

    if not data or "end" not in data:
        return jsonify({"error": "'end' field is required in request body"}), 400

    start = int(data.get("start", 0))
    end = int(data["end"])

    if start < 0 or end < start:
        return jsonify({"error": "'end' must be >= 'start' and both must be >= 0"}), 400

    try:
        predictions = model.predict(start=start, end=end)

        if hasattr(predictions, "tolist"):
            predictions = predictions.tolist()
        else:
            predictions = list(predictions)

        return jsonify({
            "start": start,
            "end": end,
            "predictions": predictions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Model summary endpoint ────────────────────────────────────────────────────
@app.route("/summary", methods=["GET"])
def summary():
    """
    GET /summary
    Returns basic model info.
    """
    info = {"model_type": type(model).__name__}

    try:
        if hasattr(model, "model"):
            inner = model.model
            if hasattr(inner, "order"):
                info["order"] = inner.order
            if hasattr(inner, "seasonal_order"):
                info["seasonal_order"] = inner.seasonal_order
            if hasattr(inner, "endog_names"):
                info["target_variable"] = str(inner.endog_names)

        if hasattr(model, "params"):
            info["params"] = model.params.to_dict() if hasattr(model.params, "to_dict") else list(model.params)

        if hasattr(model, "aic"):
            info["aic"] = model.aic
        if hasattr(model, "bic"):
            info["bic"] = model.bic

    except Exception as e:
        info["warning"] = f"Could not extract full model info: {str(e)}"

    return jsonify(info)


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)