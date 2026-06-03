# Low-Light Electro-Optic Object Tracking Pipeline

Real-time smart surveillance system that combines **low-light image enhancement** with **AI-powered object detection and tracking** for intrusion detection in restricted zones.

## Project Demo
![Pipeline Demo](assets/demo_screenshot.jpg)

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **Low-Light Image Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization) for visibility improvement in dark/low-light conditions
- **Pseudo-Thermal Imaging**: Infrared-style colormap simulation for electro-optic visualization
- **Real-Time Object Detection**: YOLOv8 nano model for fast and accurate person detection
- **Persistent Object Tracking**: ByteTrack algorithm assigns unique IDs to track individuals across frames
- **Restricted Zone Intrusion Detection**: Polygon-based virtual boundary with automated visual alerts
- **Split-Screen Display**: Side-by-side view of raw camera feed vs. AI-enhanced electro-optic feed
- **Real-Time FPS Counter**: Performance monitoring overlay

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.8+ |
| Computer Vision | OpenCV 4.x |
| Object Detection | YOLOv8 (Ultralytics) |
| Object Tracking | ByteTrack |
| Image Enhancement | CLAHE, Colormaps |
| Numerical Computing | NumPy |

## Installation

```bash
# Clone the repository
git clone https://github.com/haricharanv/Smart-Surveillance-YOLOv8.git
cd Smart-Surveillance-YOLOv8

# Install dependencies
pip install ultralytics opencv-python numpy
```

## Usage

### 1. Main Surveillance Pipeline
Run the main real-time surveillance system:
```bash
python smart_surveillance.py
```
- **Webcam Mode**: Automatically runs on camera `0`. If blocked or in use, it falls back to Demo Mode.
- **Demo Mode**: Downloads and loops a sample video `people_walking.mp4`.
- **Keyboard Controls**:
  - `C`: Cycle between sensor colormaps (*Inferno, Bone, Hot, Jet*).
  - `F`: Toggle edge fusion on/off.
  - `Z`: Toggle boundary zone visibility.
  - `Q`: Close the application.

### 2. Model Export & CPU Benchmarking (ONNX)
Export the YOLOv8 PyTorch model weights (`.pt`) to the ONNX Runtime format and benchmark execution speeds on CPU:
```bash
python export_model.py
```

### 3. Evaluation & Tracking Metrics Sandbox
Calculate Intersection over Union (IoU) between bounding boxes, compute classification metrics (Precision, Recall, F1), and read explanations of advanced tracking metrics (MOTA, IDF1):
```bash
python evaluate_metrics.py
```

### 4. C++ Inference Integration Sandbox
Build and run the C++ client to load the exported `yolov8n.onnx` model, perform image preprocessing, and parse raw prediction tensors using OpenCV's DNN module:
```bash
# Setup build directory
mkdir build && cd build
cmake ..
cmake --build . --config Release

# Run the compiled C++ inference demo
./Release/inference_demo
```

## Pipeline Architecture

```
Webcam Feed
    │
    ├──► Raw Frame (Left Display)
    │
    ▼
CLAHE Enhancement
    │
    ▼
Thermal Colormap
    │
    ▼
YOLOv8 Detection ──► ByteTrack Tracking
    │                       │
    ▼                       ▼
Restricted Zone     Persistent IDs
Check                    │
    │                       │
    ▼                       ▼
Intrusion Alert ◄── Annotated Frame (Right Display)
    │
    ▼
Split-Screen Output
```

## Configuration

Key parameters can be adjusted at the top of `smart_surveillance.py`:

| Parameter | Default | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | 0.45 | Minimum detection confidence |
| `CLAHE_CLIP_LIMIT` | 3.0 | Contrast enhancement strength |
| `THERMAL_COLORMAP` | INFERNO | Pseudo-thermal color scheme |
| `TARGET_CLASSES` | [0] (person) | COCO classes to detect |

## Applications

- Defense surveillance and perimeter security
- Critical infrastructure monitoring
- Low-light and night-vision object tracking
- Restricted area access control

## License

MIT License
