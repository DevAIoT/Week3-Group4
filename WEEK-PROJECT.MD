PROMPT (GEMINI):
Create a project description for our AIoT project:
The goal is to detect if packages in a delivery van are ok.
We will use an ESP32 with an IMU sensor to detect the VAN orientation and whether it is shaking or driving on bumpy road (accelerometer). If we detect movement we will trigger our Nvidia Jetson that has a camera to double check that a package has dropped or moved. The Jetson will compare video frames to each to check if something inside the van has moved.
Optional:
each package in the van could have a QR on the package to identify each package and the camera could read the QR of the package that has dropped. But this would require that the QR is facing the camera.
Communication between the ESP and Jetson could be wired for the prototype, but in real world it would be something wireless like BLE.

Components
ESP32 with IMU
Nvidia Jetson with Raspberry Pi camera module
   Project Title: VanGuard AI – Intelligent Cargo Stability Monitor
Project Overview
VanGuard AI is a smart monitoring system designed to detect and verify cargo shifts or drops within a delivery van in real-time. Instead of constantly recording video—which is computationally expensive and data-heavy—the system uses a low-power triggered architecture.
The system senses when the van is driving over bumpy roads or making sharp turns using an ESP32 and IMU. If the "shaking" exceeds a safe threshold, it wakes up an Nvidia Jetson "brain" to visually inspect the cargo area and confirm if any packages have actually moved or fallen.

System Architecture
The project is divided into two main layers: the Kinetic Trigger and the Visual Verifier.
1. Kinetic Trigger Layer (ESP32 + MPU6050)
* Role: Constant environmental monitoring.
* Logic: The ESP32 reads data from the 3-axis accelerometer and gyroscope. It specifically monitors for high-G impacts (bumps) or extreme tilts (sharp turns).
* Processing: It calculates the Signal Magnitude Area (SMA) or simple variance thresholds. If the van experiences a significant "jolt," the ESP32 sends a signal to the Jetson.
* Communication: * Prototype: Wired (Serial/UART).
    * Production: Bluetooth Low Energy (BLE) to allow the ESP32 to be mounted anywhere in the van chassis without running long cables.
2. Visual Verification Layer (Nvidia Jetson + Pi Cam)
* Role: Intelligent event confirmation.
* Logic: Upon receiving the trigger, the Jetson captures a burst of frames. It uses a Frame Differencing algorithm to compare the new state of the cargo hold against a "reference" frame (taken when the van was last stable).
* Movement Detection: If the pixels in specific regions of the van's interior have changed significantly, the system flags a "Cargo Displacement Event."

Technical Components
Component	Function	Technical Highlight
ESP32	Edge Sensor Node	Low power, deep-sleep capable, built-in BLE.
IMU (MPU6050)	Motion Sensing	Detects 6-axis vibration and orientation.
Nvidia Jetson	Computer Vision	High-speed processing for frame analysis.
Raspberry Pi Camera	Visual Input	Wide-angle view of the van's interior.
Optional Feature: Package Tracking (QR Codes)
To make the system "Package-Aware," each box can be labeled with a QR code.
* Inventory Mapping: The Jetson can map the position of each QR code at the start of a route.
* Specific Detection: If a movement is detected, the Jetson can identify which specific package dropped by checking which QR code changed coordinates.
* Requirement: The QR code must remain within the camera's line-of-sight and be oriented toward the lens.

Why This Matters
Traditional dashcams or cargo sensors either record too much "dead" footage or provide "blind" vibration alerts that could just be a pothole. VanGuard AI provides context-aware proof: it only alerts the driver or fleet manager when a physical bump (detected by IMU) actually results in a visual change (confirmed by AI).
