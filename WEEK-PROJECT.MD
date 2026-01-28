# Project: VanGuard AI – Integrated Cargo Intelligence System

## 1. Executive Summary
**VanGuard AI** is a compact, edge-based AIoT solution designed to monitor cargo integrity within delivery vehicles. Unlike traditional surveillance that requires constant human monitoring or massive data storage, VanGuard AI uses a **"Sense-then-See"** approach. By integrating an **IMU (Inertial Measurement Unit)** and a **High-Definition Camera** onto a single **NVIDIA Jetson** platform, the system intelligently detects physical disturbances and visually confirms if packages have shifted, fallen, or been damaged.

---

## 2. System Architecture
The system is built on a unified hardware stack, reducing latency and simplifying power management. 

### Phase A: Kinetic Monitoring (The Trigger)
The **IMU sensor** (connected via I2C) serves as the system's "inner ear." It continuously monitors the van’s dynamics:
* **Accelerometer:** Detects sudden G-force spikes caused by potholes, speed bumps, or collisions.
* **Gyroscope:** Detects extreme vehicle orientation, such as sharp turns or steep inclines that might cause stacking instability.

### Phase B: Visual Verification (The Confirmation)
When the IMU data exceeds a specific safety threshold, the **NVIDIA Jetson** triggers the **Raspberry Pi Camera Module**:
* **Frame Differencing:** The AI compares a "Pre-Event" reference frame with a "Post-Event" frame.
* **Edge Analysis:** Using OpenCV, the system calculates the percentage of pixel change within the cargo area. 
* **Intelligent Filtering:** The algorithm is tuned to ignore lighting shifts, focusing only on structural changes in the environment (i.e., a box moving from its original coordinates).

---

## 3. Hardware Components
| Component | Purpose | Technical Detail |
| :--- | :--- | :--- |
| **NVIDIA Jetson** | Central Processing Unit | Handles real-time IMU data processing and GPU-accelerated Computer Vision. |
| **IMU Sensor** | Kinetic Detection | 6-axis motion tracking to detect "unstable driving events." |
| **Pi Camera Module** | Visual Evidence | Captures high-resolution frames of the van interior. |

---

## 4. Operational Workflow
1. **Idle State:** The Jetson monitors IMU data at low power.
2. **Event Detection:** The van hits a significant bump; the IMU registers a $Z$-axis spike.
3. **Vision Activation:** The Jetson captures a video frame.
4. **Comparison:** The system compares the new frame to the "Stable State" image.
5. **Alert:** If the frame difference is high, an alert is logged: *"Potential Package Displacement Detected."*

---

## 5. Advanced Features (Optional)
### QR Code Identification
To provide granular data, packages can be equipped with **QR codes**. 
* **Inventory Mapping:** If a displacement is confirmed, the Jetson runs a QR detection script.
* **Specific Logging:** The system identifies exactly which package has moved (e.g., "Package ID: 8821").
* **Constraint:** This requires the packages to be loaded with the QR codes facing the camera lens.

---

## 6. Future Roadmap: Wireless Expansion
While the current prototype is a centralized wired unit, the architecture is designed to scale. In a production environment, the **ESP32** could be introduced as a wireless **BLE (Bluetooth Low Energy)** node. This would allow multiple IMU sensors to be placed on individual shelves, all communicating back to a single NVIDIA Jetson "Gateway" for visual confirmation.
