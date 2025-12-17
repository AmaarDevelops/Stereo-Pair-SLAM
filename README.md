# Stereo-Seed Visual SLAM (v1.0)
A Python-based Visual SLAM (Simultaneous Localization and Mapping) system. This project uses a "Stereo Seed" approach: it takes two static images to build a 3D world model, 
then tracks a live monocular webcam's movement through that 3D space.

## 🚀 Overview
Unlike standard SLAM which can be complex to initialize, this system uses a pre-calculated depth map from a stereo pair. This allows the system to know the exact
3D location of objects (landmarks) immediately upon startup.


## 🧠 Core Features
* **Stereo Depth Estimation**: Computes 3D coordinates using Block Matching (StereoBM) and reprojection matrices.
* **Feature Tracking**: Implements Lucas-Kanade (LK) Optical Flow to maintain a lock on points across frames.
* **Pose Estimation**: Uses the Perspective-n-Point (PnP) algorithm with RANSAC to calculate the camera's 6-DOF (Degrees of Freedom) trajectory.
* **Real-time Visualization**: Dynamic 3D plotting of the camera path and point cloud using Matplotlib.

## 🛠️ Tech Stack
* **Python 3.12**
* **OpenCV**: Image processing and computer vision math.
* **Matplotlib**: 3D trajectory rendering.
* **NumPy**: Linear algebra and matrix transformations.

## 📂 Project Structure
``text
.
├── slam.py            # Main application logic
├── left.PNG           # Left stereo reference image (Seed)
├── right.PNG          # Right stereo reference image (Seed)
└── README.md          # Project documentation

⚙️ Mathematical Parameters

To adapt this to your own hardware, the following constants are defined in slam.py:

Focal Length: 713.8 (Tuned for standard 720p webcams)

Baseline: 0.25 (The distance in meters between your left and right photo shots)

Principal Point: (319.5, 239.5) (Center of the image)

🚦 Getting Started


Install Dependencies:



pip install opencv-python numpy matplotlib

Run the SLAM:
python slam.py


🎮 Usage

Position your webcam so it sees the same scene as left.PNG.

Wait for the green tracking dots to appear on the "SLAM Live Feed".

Move the camera slowly. The 3D plot will begin drawing a blue line representing your real-world movement.

Press 'q' to exit the application.

Developed as a foundational project in Computer Vision and Robotics.
