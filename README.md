# GazeSonar - Tobii Glasses + YOLO Object Detection

Real-time object detection system that identifies what you're looking at using Tobii Pro Glasses 2 and YOLO.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Auto-discover Tobii Glasses on network
python gaze_sonar.py

# Specify Tobii Glasses IP address
python gaze_sonar.py --tobii-address 192.168.71.50

# Use different YOLO model
python gaze_sonar.py --yolo-model yolo11s.pt

# Save output video
python gaze_sonar.py --save-video --output my_recording.mp4
```

### Module Usage

```python
from tobii_capture import TobiiFrameCapture
from yolo_detector import YOLODetector

# Capture frames from Tobii Glasses
capture = TobiiFrameCapture()
capture.connect()
frame, gaze_position = capture.get_frame_with_gaze()

# Detect objects with YOLO
detector = YOLODetector()
detections = detector.detect_objects(frame)
gazed_object = detector.get_object_at_gaze(frame, gaze_position)
```

## Features

- Real-time video capture from Tobii Pro Glasses 2
- Gaze position extraction and visualization
- YOLO object detection
- Identification of objects at gaze position
- Gaze history tracking with smoothing
- Performance statistics
- Video recording capability

## Components

1. **tobii_capture.py**: Handles connection and data capture from Tobii Glasses
2. **yolo_detector.py**: YOLO-based object detection and gaze-object matching
3. **gaze_sonar.py**: Main application combining gaze tracking and object detection

## Requirements

- Tobii Pro Glasses 2 device
- Python 3.7+
- Network connection to Tobii Glasses