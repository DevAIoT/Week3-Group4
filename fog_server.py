import ssl
import certifi
import numpy as np
import requests
from flask import Flask, request, jsonify
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Fix SSL certificate verification issue
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize Flask app
app = Flask(__name__)

# Global model variable - initialized once at startup
model = None

# Configuration - Cloud server IP (replace with actual Cloud server IP)
CLOUD_SERVER_IP = "10.158.13.201"  # Update this with Cloud server's actual IP
CLOUD_PREDICT_URL = f"http://{CLOUD_SERVER_IP}:5000/predict"

# Variance threshold for routing decision
# Lower variance = simpler image -> process locally
# Higher variance = complex image -> forward to Cloud
VARIANCE_THRESHOLD = 3500.0  # Edge uses 2000.0 (calculated on raw images 0-255 range)

# Statistics tracking
stats = {
    "local_inferences": 0,
    "cloud_forwards": 0
}


def load_model():
    """Load MobileNetV2 model once at startup"""
    print("Loading MobileNetV2 model on Fog node...")
    model = MobileNetV2(weights="imagenet", include_top=True, alpha=1.0)
    print("Model loaded successfully on Fog!")
    return model


def calculate_image_complexity(image_array):
    """
    Calculate complexity of image using variance of pixel values
    Higher variance = more complex/diverse image

    Args:
        image_array: numpy array of shape (1, 224, 224, 3)

    Returns:
        float: variance value representing image complexity
    """
    return np.var(image_array)


def forward_to_cloud(image_array):
    """
    Forward the image to Cloud server for inference

    Args:
        image_array: numpy array of shape (1, 224, 224, 3)

    Returns:
        dict: prediction result from Cloud
    """
    try:
        # Convert numpy array to Python list for JSON serialization
        image_list = image_array.tolist()

        # Prepare JSON payload
        payload = {"image": image_list}

        # Send POST request to Cloud server
        response = requests.post(CLOUD_PREDICT_URL, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            # Add inference source
            result["inference_source"] = "cloud"
            stats["cloud_forwards"] += 1
            return result
        else:
            return {"error": f"Cloud server error: {response.text}"}

    except Exception as e:
        return {"error": f"Failed to reach Cloud: {str(e)}"}


def local_inference(image_array):
    """
    Perform inference locally on Fog node

    Args:
        image_array: numpy array of shape (1, 224, 224, 3)

    Returns:
        dict: prediction result
    """
    # Perform prediction
    predictions = model.predict(image_array, verbose=0)

    # Get confidence scores (top-5 predictions)
    top_indices = np.argsort(predictions[0])[-5:][::-1]
    confidence_scores = predictions[0][top_indices].tolist()

    # Track local inference
    stats["local_inferences"] += 1

    # Return confidence scores and top prediction
    response = {
        "top_confidence": float(predictions[0].max()),
        "top_5_confidences": confidence_scores,
        "top_5_classes": top_indices.tolist(),
        "all_predictions": predictions[0].tolist(),
        "inference_source": "fog"
    }

    return response


@app.route("/predict", methods=["POST"])
def predict():
    """
    Fog node prediction endpoint with smart routing
    - Uses complexity value from Edge (calculated on raw image)
    - Routes to local inference or Cloud based on threshold
    """
    try:
        # Extract JSON data from request
        data = request.get_json()

        if "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Convert list back to numpy array
        image_data = np.array(data["image"], dtype="float32")

        # Ensure proper shape (should be (1, 224, 224, 3))
        if image_data.shape != (1, 224, 224, 3):
            return jsonify({
                "error": f"Invalid image shape: {image_data.shape}. Expected (1, 224, 224, 3)"
            }), 400

        # Get complexity value from payload (calculated by Edge on raw image)
        # If not present, calculate it on the preprocessed image (fallback)
        if "complexity" in data:
            complexity = float(data["complexity"])
        else:
            complexity = calculate_image_complexity(image_data)

        print(f"Image complexity: {complexity:.2f} | Threshold: {VARIANCE_THRESHOLD}")

        # Smart routing decision
        if complexity < VARIANCE_THRESHOLD:
            # Simple image - process locally on Fog
            print("→ Processing locally on Fog (low complexity)")
            result = local_inference(image_data)
            result["complexity"] = float(complexity)
            return jsonify(result), 200
        else:
            # Complex image - forward to Cloud
            print("→ Forwarding to Cloud (high complexity)")
            result = forward_to_cloud(image_data)
            result["complexity"] = float(complexity)
            return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "node_type": "fog",
        "model_loaded": model is not None,
        "cloud_server": CLOUD_SERVER_IP
    }), 200


@app.route("/stats", methods=["GET"])
def get_stats():
    """Return statistics about routing decisions"""
    total = stats["local_inferences"] + stats["cloud_forwards"]
    return jsonify({
        "local_inferences": stats["local_inferences"],
        "cloud_forwards": stats["cloud_forwards"],
        "total_requests": total,
        "local_percentage": (stats["local_inferences"] / total * 100) if total > 0 else 0,
        "cloud_percentage": (stats["cloud_forwards"] / total * 100) if total > 0 else 0
    }), 200


if __name__ == "__main__":
    # Initialize model once at startup
    model = load_model()

    # Run Flask server
    print("\n" + "=" * 60)
    print("FOG NODE - Server + Client Hybrid")
    print("=" * 60)
    print(f"Fog Server running on http://0.0.0.0:5000")
    print(f"Cloud Server endpoint: {CLOUD_PREDICT_URL}")
    print(f"Variance Threshold: {VARIANCE_THRESHOLD}")
    print(f"  - Complexity < {VARIANCE_THRESHOLD}: Process locally on Fog")
    print(f"  - Complexity >= {VARIANCE_THRESHOLD}: Forward to Cloud")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5000, debug=False)
