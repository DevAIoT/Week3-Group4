import time
import psutil
import os
import numpy as np
import requests
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Configuration: Server endpoint
SERVER_URL = "http://localhost:5000/predict"


# Function to generate a dummy image
def generate_dummy_image(shape=(224, 224, 3)):
    """
    Generate a random dummy image
    Returns a batch of size 1: (1, 224, 224, 3)
    """
    # Generate random float values between 0 and 255
    image = np.random.randint(0, 255, shape).astype("float32")
    # Preprocess input as expected by MobileNetV2
    image = preprocess_input(image)
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image


# Function to monitor resources
def get_resource_usage():
    """Get current CPU and memory usage"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    # RSS (Resident Set Size) is the non-swapped physical memory a process has used
    mem_usage_mb = mem_info.rss / 1024 / 1024
    cpu_usage = psutil.cpu_percent(interval=None)
    return cpu_usage, mem_usage_mb


def send_prediction_request(image_array):
    """
    Convert image to Python list and send to server
    Returns response time and prediction result
    """
    # Convert numpy array to Python list for JSON serialization
    image_list = image_array.tolist()

    # Prepare JSON payload
    payload = {"image": image_list}

    # Record start time
    request_start = time.time()

    # Send POST request to server
    response = requests.post(SERVER_URL, json=payload)

    # Record end time
    request_end = time.time()

    # Calculate request-response time
    response_time = request_end - request_start

    # Parse response
    if response.status_code == 200:
        result = response.json()
        return response_time, result
    else:
        return response_time, {"error": response.text}


def run_client_benchmark(mode="same", iterations=100):
    """
    Main client loop to send images to server and measure performance

    Args:
        mode: 'same' to send the same image, 'different' to generate new images each time
        iterations: number of requests to send
    """
    print(f"\n=== Client Benchmark: {mode.upper()} Images ===")
    print(f"Server endpoint: {SERVER_URL}")
    print(f"Iterations: {iterations}")

    # Pre-generate one image for 'same' mode
    static_image = generate_dummy_image()

    # Warm-up request
    print("\nWarming up with initial request...")
    try:
        _, warmup_result = send_prediction_request(static_image)
        print(
            f"Warmup successful. Top confidence: {warmup_result.get('top_confidence', 'N/A')}"
        )
    except Exception as e:
        print(f"Error during warmup: {e}")
        print("Make sure the server is running!")
        return

    # Initialize metrics
    response_times = []
    total_start = time.time()
    cpu_start, mem_start = get_resource_usage()

    print(f"\nStarting {iterations} requests...")

    for i in range(iterations):
        if mode == "same":
            # Use the already generated static image
            img = static_image
        else:
            # Generate a new random image for every iteration
            img = generate_dummy_image()

        try:
            # Send request and measure time
            resp_time, result = send_prediction_request(img)
            response_times.append(resp_time)

            # Print progress every 10 iterations
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{iterations} requests completed")

        except Exception as e:
            print(f"Error on request {i + 1}: {e}")
            continue

    # Calculate final metrics
    total_end = time.time()
    cpu_end, mem_end = get_resource_usage()

    total_time = total_end - total_start
    avg_response_time = np.mean(response_times)
    min_response_time = np.min(response_times)
    max_response_time = np.max(response_times)
    std_response_time = np.std(response_times)

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Mode: {mode.upper()}")
    print(f"Total Requests: {iterations}")
    print(f"Successful Requests: {len(response_times)}")
    print()
    print("TIME METRICS:")
    print(f"  Total Time: {total_time:.4f} sec")
    print(f"  Avg Response Time: {avg_response_time:.4f} sec")
    print(f"  Min Response Time: {min_response_time:.4f} sec")
    print(f"  Max Response Time: {max_response_time:.4f} sec")
    print(f"  Std Response Time: {std_response_time:.4f} sec")
    print()
    print("RESOURCE USAGE (CLIENT):")
    print(f"  Memory Start: {mem_start:.2f} MB")
    print(f"  Memory End: {mem_end:.2f} MB")
    print(f"  Memory Delta: {mem_end - mem_start:.2f} MB")
    print(f"  CPU Usage (Snapshot): {cpu_end:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MOBILENETV2 CLIENT - EDGE DEVICE")
    print("=" * 60)

    # Check if server is reachable
    print("\nChecking server availability...")
    try:
        health_check = requests.get("http://localhost:5000/health", timeout=2)
        if health_check.status_code == 200:
            print("✓ Server is running and healthy")
        else:
            print("✗ Server returned unexpected status")
    except requests.exceptions.RequestException:
        print("✗ Cannot reach server. Please start server.py first!")
        print("  Run: python server.py")
        exit(1)

    # Scenario A: Same image for all predictions
    run_client_benchmark(mode="same", iterations=100)

    # Wait a bit between scenarios
    time.sleep(2)

    # Scenario B: Different image for each prediction
    run_client_benchmark(mode="different", iterations=100)
