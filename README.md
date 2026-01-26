# Week3-Group4

---

## **Task 1: Vertical Scaling Evaluation Report**

### **1. Objective**

The primary goal of this experiment was to evaluate the impact of hardware configuration and power management on the inference performance of a Deep Learning model (**MobileNetV2**) running locally at the **Edge**.

### **2. Methodology**

We developed a Python script utilizing **TensorFlow/Keras** and **NumPy** to perform the following:

* **Model Initialization:** Loaded MobileNetV2 with pre-trained weights.
* **Data Generation:** Generated dummy input tensors of shape .
* **Inference Loop:** Executed 100 consecutive predictions.
* **Test Scenarios:** 1.  **Uniform Input:** Same images across all test iterations.
2.  **Diverse Input:** Different random images for each iteration.
* **Resource Monitoring:** Captured CPU/RAM utilization and precise execution latency (Start/End times).

---

### **3. Experimental Results**

Based on our testing across the group's hardware, we observed a significant performance delta between power modes:

| Metric | Power Saver Mode (Unplugged) | High Performance Mode (Plugged In) |
| --- | --- | --- |
| **Inference Latency** | Higher (Slower) | Lower (Faster) |
| **CPU Clock Speed** | Throttled / Scaled Down | Maximum Performance |
| **Consistency** | High variability due to power saving | Stable and Rapid |

**Key Finding:** When connected to a power source, the CPU is permitted to operate at its maximum frequency without thermal or battery-saving constraints. This directly translates to faster mathematical computations required for the neural network's layers.

---

### **4. Analysis and AIoT Implications**

#### **Vertical Scaling & Hardware Capability**

Vertical scaling in this context refers to optimizing the individual "Edge" node. Our results show that hardware configuration (Power Profile) is just as critical as raw specs.

* **High Performance Mode:** Ideal for real-time AIoT applications (e.g., surveillance or industrial defect detection) where low latency is mandatory.
* **Power Saver Mode:** Suitable for non-critical periodic monitoring where battery longevity is prioritized over speed.

#### **Model Selection: MobileNetV2 vs. Others**

* **MobileNetV2** is highly optimized for Edge devices using depthwise separable convolutions.
* **Heavier Models (e.g., ResNet50, VGG16):** If we utilized larger models, the performance gap between "Power Saver" and "High Performance" would likely widen. Heavier models require more FLOPs (Floating Point Operations), making them even more sensitive to CPU throttling and memory bandwidth limits.

---

### **5. Visual Evidence**

The following images document the resource consumption and execution logs during the test:

* **![Image 1: System Monitoring during Power Saver Mode](SCR-20260126-kpwu.png)**
* **![Image 2: System Monitoring during High Performance Mode](SCR-20260126-kmti-2.png)**


## **Task 2**

* **![Image 3: Off load](SCR-20260126-kvwb.png)**

---
