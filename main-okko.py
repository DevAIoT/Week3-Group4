import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import time
import psutil
import os
import ssl

# Workaround for SSL certificate errors on some macOS Python installations
# when downloading model weights
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- Helper: Resource Monitoring ---
def get_resource_usage():
    """Returns current memory usage (MB) and CPU percent."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # Convert bytes to MB
    cpu = psutil.cpu_percent(interval=None) # Non-blocking CPU check
    return mem, cpu

# --- Step 1: Initialize the MobileNetV2 model ---
print("Initializing MobileNetV2...")
# We use weights='imagenet' for a standard initialized model
model = MobileNetV2(weights='imagenet')

# *Warmup step* (Optional but recommended): 
# The first inference is always slower due to graph construction. 
# We run one dummy prediction so it doesn't skew our timing results.
dummy_warmup = np.zeros((1, 224, 224, 3))
model.predict(dummy_warmup, verbose=0)
print("Model initialized and warmed up.\n")

# --- Step 2: Function to generate dummy image ---
def generate_dummy_image(batch_size=1):
    """
    Generates a random NumPy array of shape (batch_size, 224, 224, 3).
    Preprocesses it to fit MobileNetV2 requirements.
    """
    # Generate random noise between 0 and 255
    random_img = np.random.randint(0, 255, (batch_size, 224, 224, 3)).astype('float32')
    # Preprocess (scales input to [-1, 1] for MobileNetV2)
    return preprocess_input(random_img)

# --- Step 3 & 4: Evaluation Loop & Recording Resources ---
def run_evaluation(mode="same"):
    print(f"--- Starting Evaluation: {mode.upper()} IMAGE(S) ---")
    
    # Prepare data based on mode
    if mode == "same":
        # Generate ONE image and reuse it 100 times
        input_data = generate_dummy_image()
        dataset = [input_data for _ in range(100)]
    else:
        # Generate 100 DIFFERENT images
        dataset = [generate_dummy_image() for _ in range(100)]

    # Snapshot resources before starting
    mem_before, _ = get_resource_usage()
    
    # Start Timer
    start_time = time.time()
    
    # Prediction Loop (100 iterations)
    for img in dataset:
        model.predict(img, verbose=0)
        
    # End Timer
    end_time = time.time()
    
    # Snapshot resources after
    mem_after, _ = get_resource_usage()
    
    # Calculate stats
    total_time = end_time - start_time
    avg_time_per_img = total_time / 100
    mem_diff = mem_after - mem_before
    
    print(f"Total Time: {total_time:.4f} seconds")
    print(f"Avg Time per Image: {avg_time_per_img:.4f} seconds")
    print(f"Memory Usage Change: {mem_diff:.2f} MB")
    print("-" * 40 + "\n")

# Run Scenario A: Same images for all clients (iterations)
run_evaluation(mode="same")

# Run Scenario B: Different images for each client (iterations)
run_evaluation(mode="different")