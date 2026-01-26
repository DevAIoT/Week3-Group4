import ssl
import certifi
import requests
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Fix SSL certificate verification issue
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize Flask app
app = Flask(__name__)

# Global model variable - initialized once at startup
model = None


def load_model():
    """Load MobileNetV2 model once at startup"""
    print("Loading MobileNetV2 model...")
    model = MobileNetV2(weights="imagenet", include_top=True, alpha=1.0)
    print("Model loaded successfully!")
    return model


FOG_THRESHOLD = 500  # ถ้า Variance เกินค่านี้ จะส่งต่อให้ Cloud
CLOUD_URL = "http://localhost:5001/predict"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_data = np.array(data["image"], dtype="float32")

    # คำนวณ Variance เพื่อตัดสินใจ
    pixel_variance = np.var(image_data)

    # LOGIC: ถ้าภาพซับซ้อนมาก (Variance สูง) ให้ส่งต่อ Cloud
    if pixel_variance > FOG_THRESHOLD:
        print(
            f"Fog: Logic high variance ({pixel_variance:.2f}), Offloading to Cloud..."
        )
        response = requests.post(CLOUD_URL, json=data)
        result = response.json()
        result["inference_source"] = "Cloud (via Fog)"
        return jsonify(result)

    # ถ้าไม่เกิน ให้ประมวลผลที่ Fog เอง
    print(f"Fog: Processing locally (variance: {pixel_variance:.2f})")
    predictions = model.predict(image_data, verbose=0)

    # Get confidence scores (top-5 predictions)
    top_indices = np.argsort(predictions[0])[-5:][::-1]
    confidence_scores = predictions[0][top_indices].tolist()

    response_data = {
        "top_confidence": float(predictions[0].max()),
        "top_5_confidences": confidence_scores,
        "top_5_classes": top_indices.tolist(),
        "all_predictions": predictions[0].tolist(),
        "inference_source": "Fog",
    }
    return jsonify(response_data)


if __name__ == "__main__":
    # Initialize model once at startup
    model = load_model()

    # Run Flask server
    print("\n=== Starting Server ===")
    print("Server running on http://localhost:5002")
    print("Endpoint: POST http://localhost:5002/predict")

    app.run(host="0.0.0.0", port=5002, debug=False)
