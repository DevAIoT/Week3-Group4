import time
import ssl
import certifi
import psutil
import os
import numpy as np
import requests
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Fix SSL certificate verification issue
ssl._create_default_https_context = ssl._create_unverified_context

# Configuration: Fog server endpoint
FOG_SERVER_IP = "10.158.13.179"  # Update with Fog server's IP
FOG_PREDICT_URL = f"http://{FOG_SERVER_IP}:5000/predict"
FOG_HEALTH_URL = f"http://{FOG_SERVER_IP}:5000/health"

# Variance threshold for Edge routing decision
# Edge has lower threshold - processes only very simple images locally
# More complex images are forwarded to Fog/Cloud
EDGE_VARIANCE_THRESHOLD = 2000.0  # Lower than Fog's 5000.0

# Global model variable
model = None

# Statistics tracking
stats = {
    "edge_local": 0,
    "fog_processed": 0,
    "cloud_processed": 0,
    "total_requests": 0
}

# Latency tracking by source
latency_by_source = {
    "edge": [],
    "fog": [],
    "cloud": []
}


def load_model():
    """Load MobileNetV2 model for Edge processing"""
    print("Loading MobileNetV2 model on Edge device...")
    model = MobileNetV2(weights="imagenet", include_top=True, alpha=1.0)
    print("Model loaded successfully on Edge!")
    return model


def generate_simple_image(shape=(224, 224, 3)):
    """
    Generate a simple image with low variance (uniform color with slight noise)
    """
    # Create mostly uniform color with small random variations
    base_color = np.random.randint(50, 200)
    raw_image = np.full(shape, base_color, dtype="float32")
    # Add small noise
    noise = np.random.randint(-20, 20, shape).astype("float32")
    raw_image = np.clip(raw_image + noise, 0, 255)
    return raw_image


def generate_medium_image(shape=(224, 224, 3)):
    """
    Generate a medium complexity image with moderate variance (gradients + patterns)
    """
    raw_image = np.zeros(shape, dtype="float32")

    # Create gradient patterns
    for c in range(3):
        gradient = np.linspace(50, 200, shape[0])
        raw_image[:, :, c] = gradient[:, np.newaxis]

    # Add moderate random noise
    noise = np.random.randint(-50, 50, shape).astype("float32")
    raw_image = np.clip(raw_image + noise, 0, 255)
    return raw_image


def generate_complex_image(shape=(224, 224, 3)):
    """
    Generate a complex image with high variance (random noise, detailed patterns)
    """
    # Generate fully random image with high variance
    raw_image = np.random.randint(0, 255, shape).astype("float32")
    return raw_image


def generate_dummy_image(shape=(224, 224, 3)):
    """
    Generate a random dummy image with varying complexity
    Randomly selects between simple, medium, or complex image types
    Returns raw image (for variance calculation) and preprocessed image (for inference)
    """
    # Randomly choose image type
    image_type = np.random.choice(["simple", "medium", "complex"])

    if image_type == "simple":
        raw_image = generate_simple_image(shape)
    elif image_type == "medium":
        raw_image = generate_medium_image(shape)
    else:  # complex
        raw_image = generate_complex_image(shape)

    # Create preprocessed version for inference
    preprocessed_image = preprocess_input(raw_image.copy())

    # Add batch dimension to preprocessed image
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

    return raw_image, preprocessed_image


def get_resource_usage():
    """Get current CPU and memory usage"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_usage_mb = mem_info.rss / 1024 / 1024
    cpu_usage = psutil.cpu_percent(interval=None)
    return cpu_usage, mem_usage_mb


def calculate_image_complexity(raw_image):
    """
    Calculate complexity of image using variance of pixel values on RAW image (0-255 range)
    Higher variance = more complex/diverse image

    Args:
        raw_image: numpy array of raw image data (224, 224, 3) in 0-255 range

    Returns:
        float: variance value
    """
    return np.var(raw_image)


def local_inference(image_array):
    """
    Perform inference locally on Edge device

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
    stats["edge_local"] += 1

    # Return confidence scores and top prediction
    response = {
        "top_confidence": float(predictions[0].max()),
        "top_5_confidences": confidence_scores,
        "top_5_classes": top_indices.tolist(),
        "all_predictions": predictions[0].tolist(),
        "inference_source": "edge"
    }

    return response


def forward_to_fog(raw_image, complexity):
    """
    Forward image to Fog server for inference
    Sends preprocessed image along with pre-calculated complexity

    Args:
        raw_image: numpy array of raw image (224, 224, 3) in 0-255 range
        complexity: pre-calculated variance from raw image

    Returns:
        tuple: (response_time, result)
    """
    try:
        # Preprocess the image before sending (for inference)
        preprocessed = preprocess_input(raw_image.copy())
        preprocessed = np.expand_dims(preprocessed, axis=0)

        # Convert numpy array to Python list for JSON serialization
        image_list = preprocessed.tolist()

        # Prepare JSON payload with image and complexity
        payload = {
            "image": image_list,
            "complexity": float(complexity)
        }

        # Record start time
        request_start = time.time()

        # Send POST request to Fog server
        response = requests.post(FOG_PREDICT_URL, json=payload, timeout=30)

        # Record end time
        request_end = time.time()
        response_time = request_end - request_start

        if response.status_code == 200:
            result = response.json()
            # Track based on where inference actually happened
            source = result.get("inference_source", "fog")
            if source == "fog":
                stats["fog_processed"] += 1
            elif source == "cloud":
                stats["cloud_processed"] += 1

            return response_time, result
        else:
            return response_time, {"error": response.text}

    except Exception as e:
        return 0, {"error": f"Failed to reach Fog: {str(e)}"}


def smart_inference(raw_image, preprocessed_image):
    """
    Smart routing decision for Edge device
    - Low complexity: Process locally
    - High complexity: Forward to Fog (which may forward to Cloud)

    Args:
        raw_image: raw image data (224, 224, 3) for complexity calculation
        preprocessed_image: preprocessed image (1, 224, 224, 3) for inference

    Returns:
        tuple: (inference_time, result)
    """
    # Calculate image complexity on RAW image (before preprocessing)
    complexity = calculate_image_complexity(raw_image)

    if complexity < EDGE_VARIANCE_THRESHOLD:
        # Very simple image - process locally on Edge
        print(f"  Complexity: {complexity:.2f} → Processing on EDGE")
        start_time = time.time()
        result = local_inference(preprocessed_image)
        inference_time = time.time() - start_time
        result["complexity"] = float(complexity)
        return inference_time, result
    else:
        # Complex image - forward to Fog (may go to Cloud)
        print(f"  Complexity: {complexity:.2f} → Forwarding to FOG/CLOUD")
        inference_time, result = forward_to_fog(raw_image, complexity)
        if "complexity" not in result:
            result["complexity"] = float(complexity)
        return inference_time, result


def run_edge_benchmark(mode="different", iterations=50):
    """
    Run Edge client benchmark with smart routing

    Args:
        mode: 'same' to use same image, 'different' to generate new images
        iterations: number of requests (minimum 50 as per task)
    """
    print("\n" + "=" * 70)
    print(f"EDGE CLIENT BENCHMARK - {mode.upper()} Images")
    print("=" * 70)
    print(f"Fog Server endpoint: {FOG_PREDICT_URL}")
    print(f"Edge Variance Threshold: {EDGE_VARIANCE_THRESHOLD}")
    print(f"Iterations: {iterations}")
    print("=" * 70)

    # Reset statistics
    stats["edge_local"] = 0
    stats["fog_processed"] = 0
    stats["cloud_processed"] = 0
    stats["total_requests"] = 0
    latency_by_source["edge"] = []
    latency_by_source["fog"] = []
    latency_by_source["cloud"] = []

    # Pre-generate one image for 'same' mode
    static_raw_image, static_preprocessed_image = generate_dummy_image()

    # Warm-up
    print("\nWarming up...")
    _, warmup_result = smart_inference(static_raw_image, static_preprocessed_image)
    print(f"Warmup complete. Source: {warmup_result.get('inference_source', 'N/A')}")

    # Initialize metrics
    all_response_times = []
    total_start = time.time()
    cpu_start, mem_start = get_resource_usage()

    print(f"\nStarting {iterations} inference requests...\n")

    for i in range(iterations):
        if mode == "same":
            raw_img = static_raw_image
            prep_img = static_preprocessed_image
        else:
            raw_img, prep_img = generate_dummy_image()

        try:
            # Perform smart inference
            resp_time, result = smart_inference(raw_img, prep_img)
            all_response_times.append(resp_time)
            stats["total_requests"] += 1

            # Track latency by source
            source = result.get("inference_source", "unknown")
            if source in latency_by_source:
                latency_by_source[source].append(resp_time)

            # Print progress every 10 iterations
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{iterations} requests completed")

        except Exception as e:
            print(f"Error on request {i + 1}: {e}")
            continue

    # Calculate final metrics
    total_end = time.time()
    cpu_end, mem_end = get_resource_usage()

    total_time = total_end - total_start

    # Print comprehensive results
    print("\n" + "=" * 70)
    print("EDGE CLIENT BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Mode: {mode.upper()}")
    print(f"Total Requests: {stats['total_requests']}")
    print()

    print("INFERENCE SOURCE DISTRIBUTION:")
    print(f"  Edge (Local):     {stats['edge_local']:3d} requests ({stats['edge_local']/stats['total_requests']*100:5.1f}%)")
    print(f"  Fog (Offloaded):  {stats['fog_processed']:3d} requests ({stats['fog_processed']/stats['total_requests']*100:5.1f}%)")
    print(f"  Cloud (Offloaded): {stats['cloud_processed']:3d} requests ({stats['cloud_processed']/stats['total_requests']*100:5.1f}%)")
    print()

    print("LATENCY STATISTICS BY SOURCE:")
    for source in ["edge", "fog", "cloud"]:
        if latency_by_source[source]:
            latencies = latency_by_source[source]
            print(f"  {source.upper()}:")
            print(f"    Count: {len(latencies)}")
            print(f"    Avg:   {np.mean(latencies):.4f} sec")
            print(f"    Min:   {np.min(latencies):.4f} sec")
            print(f"    Max:   {np.max(latencies):.4f} sec")
            print(f"    Std:   {np.std(latencies):.4f} sec")
        else:
            print(f"  {source.upper()}: No requests")
    print()

    print("OVERALL TIME METRICS:")
    print(f"  Total Time: {total_time:.4f} sec")
    print(f"  Avg Response Time: {np.mean(all_response_times):.4f} sec")
    print(f"  Min Response Time: {np.min(all_response_times):.4f} sec")
    print(f"  Max Response Time: {np.max(all_response_times):.4f} sec")
    print(f"  Std Response Time: {np.std(all_response_times):.4f} sec")
    print()

    print("RESOURCE USAGE (EDGE CLIENT):")
    print(f"  Memory Start: {mem_start:.2f} MB")
    print(f"  Memory End:   {mem_end:.2f} MB")
    print(f"  Memory Delta: {mem_end - mem_start:.2f} MB")
    print(f"  CPU Usage:    {cpu_end:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MOBILENETV2 EDGE CLIENT - Smart Routing to Fog/Cloud")
    print("=" * 70)

    # Load model for local processing
    model = load_model()

    # Check if Fog server is reachable
    print("\nChecking Fog server availability...")
    try:
        health_check = requests.get(FOG_HEALTH_URL, timeout=2)
        if health_check.status_code == 200:
            print("✓ Fog server is running and healthy")
        else:
            print("✗ Fog server returned unexpected status")
    except requests.exceptions.RequestException:
        print("✗ Cannot reach Fog server. Please start fog_server.py first!")
        print("  Run: python fog_server.py")
        exit(1)

    # Run benchmark with at least 50 requests (as per task requirement)
    run_edge_benchmark(mode="different", iterations=50)

    # Optional: Run with same image mode
    print("\n")
    time.sleep(2)
    run_edge_benchmark(mode="same", iterations=50)
