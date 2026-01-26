import time
import psutil
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. Initialize MobileNetV2
# We use 'imagenet' weights for a realistic setup.
def load_model():
    print("Loading MobileNetV2 model...")
    model = MobileNetV2(weights='imagenet', include_top=True,alpha=1.0)
    return model

# 2. Function to generate a dummy image
# Returns a batch of size 1: (1, 224, 224, 3)
def generate_dummy_image(shape=(224, 224, 3)):
    # Generate random float values between 0 and 255
    image = np.random.randint(0, 255, shape).astype('float32')
    # Preprocess input as expected by MobileNetV2
    image = preprocess_input(image)
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

# Function to monitor resources
def get_resource_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    # RSS (Resident Set Size) is the non-swapped physical memory a process has used
    mem_usage_mb = mem_info.rss / 1024 / 1024 
    cpu_usage = psutil.cpu_percent(interval=None)
    return cpu_usage, mem_usage_mb

# 3. Main benchmarking loop
def run_benchmark(model, mode='same', iterations=100):
    print(f"\n--- Starting Benchmark: {mode.upper()} Images ---")
    
    # Pre-generate one image for the 'same' mode
    static_image = generate_dummy_image()
    
    # Warm-up prediction (to compile the graph and load cuDNN if using GPU)
    # This ensures the first slow run doesn't skew our metrics
    model.predict(static_image, verbose=0)
    
    start_time = time.time()
    cpu_start, mem_start = get_resource_usage()
    
    for i in range(iterations):
        if mode == 'same':
            # Use the already generated static image
            img = static_image
        else:
            # Generate a new random image for every iteration
            img = generate_dummy_image()
            
        model.predict(img, verbose=0)

    end_time = time.time()
    cpu_end, mem_end = get_resource_usage()
    
    total_time = end_time - start_time
    avg_time = total_time / iterations
    
    print(f"Total Time: {total_time:.4f} sec")
    print(f"Avg Time per Image: {avg_time:.4f} sec")
    print(f"Memory Usage: ~{mem_end:.2f} MB (End)")
    print(f"CPU Usage (Snapshot): {cpu_end}%")

if __name__ == "__main__":
    # Initialize Model
    model = load_model()
    
    # Scenario A: Same image for all 100 predictions
    # This simulates a client sending the exact same data repeatedly (or caching).
    run_benchmark(model, mode='same', iterations=100)
    
    # Scenario B: Different image for each prediction
    # This simulates real-world traffic where every user sends a unique photo.
    run_benchmark(model, mode='different', iterations=100)
