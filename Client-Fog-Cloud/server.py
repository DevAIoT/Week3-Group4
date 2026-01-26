import ssl
import certifi
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


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint to receive image data and return prediction confidence scores
    Expects JSON with 'image' key containing the image as a Python list
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
            return (
                jsonify(
                    {
                        "error": f"Invalid image shape: {image_data.shape}. Expected (1, 224, 224, 3)"
                    }
                ),
                400,
            )

        # Perform prediction
        predictions = model.predict(image_data, verbose=0)

        # Get confidence scores (top-5 predictions)
        top_indices = np.argsort(predictions[0])[-5:][::-1]
        confidence_scores = predictions[0][top_indices].tolist()

        # Return confidence scores and top prediction
        response = {
            "top_confidence": float(predictions[0].max()),
            "top_5_confidences": confidence_scores,
            "top_5_classes": top_indices.tolist(),
            "all_predictions": predictions[0].tolist(),
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": model is not None}), 200


if __name__ == "__main__":
    # Initialize model once at startup
    model = load_model()

    # Run Flask server
    print("\n=== Starting Server ===")
    print("Server running on http://localhost:5001")
    print("Endpoint: POST http://localhost:5001/predict")

    app.run(host="0.0.0.0", port=5001, debug=False)
